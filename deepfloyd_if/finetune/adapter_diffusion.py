from pathlib import Path
import logging, sys, argparse, itertools, math

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import get_scheduler
import bitsandbytes as bnb
from tqdm import tqdm

from deepfloyd_if.modules import IFStageI, T5Embedder
from deepfloyd_if.finetune.utils import freeze_params
from deepfloyd_if.finetune.adapter import inject_adapter
from deepfloyd_if.finetune.custom_diffusion import define_args, \
    CustomDiffusionDataset, collate_fn, save_ckpt, prepare_class_images, \
    add_vtokens


def define_adapter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bottleneck_r", type=int, default=2, required=False, help="adapter bottleneck rank")
    parser.add_argument("--bottleneck_channels", type=int, default=-1, required=False, help="bottleneck channels")
    parser.add_argument("--adapter_scale", type=float, default=1.0, required=False, 
                        help="adapter scale, controling the strength of the adapter")
    parser.add_argument("--share_adapter", action='store_true', help="whether to share adapter in each down/up block")


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)

logger = logging.getLogger(__name__)


def main(args) -> None:
    set_seed(args.seed)

    t5 = T5Embedder(device=args.device, torch_dtype=torch.float32, use_offload_folder=None)
    if args.use_gradient_checkpoint:
        t5.model.gradient_checkpointing_enable()
    if_I = IFStageI(
        args.if_I, 
        device=args.device, 
        model_kwargs={
            "precision": 32, 
            "use_checkpoint": args.use_gradient_checkpoint,
            "checkpoint_use_reentrant": False
        }
    )
    # https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html#checkpoint
    if_I.model.to(dtype=if_I.model.dtype)

    prepare_class_images(
        class_data_root=args.class_data_root,
        num_class_images=args.num_class_images,
        class_prompt=args.class_prompt,
        t5=t5, if_I=if_I
    )

    # Add instance token
    vtoken_seq, vtoken_ids = add_vtokens(t5, num_tokens=args.num_vtokens, init_token=args.init_token)
    instance_prompt = args.instance_prompt.format(vtoken_seq)
    logger.info(f"Instance prompt: {instance_prompt}")
    logger.info(f"Class prompt: {args.class_prompt}")

    dataset = CustomDiffusionDataset(
        instance_data_root=args.instance_data_root,
        instance_prompt=instance_prompt,
        tokenizer=t5.tokenizer,
        class_data_root=args.class_data_root,
        class_prompt=args.class_prompt,
    )
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # optimizers
    lr = args.lr
    if args.scale_lr:
        lr = lr * args.batch_size * args.gradient_accumulation_steps

    #  torch.optim.AdamW. Here, use 8bit Adam
    adamw_opt = bnb.optim.Adam8bit if args.use_8bitadam else torch.optim.AdamW

    t5.model = t5.model.train()
    freeze_params(t5.model.encoder.parameters())
    for param in t5.model.get_input_embeddings().parameters():
        param.requires_grad = True

    if_I.model.train()
    freeze_params(if_I.model.parameters())
    if_I_trainable_parameters = inject_adapter(
        if_I.model,
        bottleneck_r=args.bottleneck_r,
        bottleneck_channels=args.bottleneck_channels,
        adapter_scale=args.adapter_scale,
        share_adapter=args.share_adapter
    )
    print(if_I_trainable_parameters)
    optimizer = adamw_opt(
        itertools.chain(*if_I_trainable_parameters, t5.model.get_input_embeddings().parameters()), # only optimize the embeddings
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08
    )

    # learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles * args.gradient_accumulation_steps,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
        log_with=args.report_to,
    )
    accelerator.init_trackers("deepfloyd-if", config=vars(args))
    t5.model, if_I.model, optimizer, data_loader, lr_scheduler = accelerator.prepare(
        t5.model, if_I.model, optimizer, data_loader, lr_scheduler
    )

    total_batch_size = args.batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Learning rate = {lr}")

    num_update_steps_per_epoch = math.ceil(len(data_loader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    diffusion = if_I.get_diffusion(timestep_respacing='')
    num_timesteps = diffusion.num_timesteps
    if_I_kwargs = {
        'guidance_scale': 7.0
    }

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(t5.model).get_input_embeddings().weight.data.clone()

    for epoch in range(num_train_epochs):
        t5.model.train()
        if_I.model.train()
        for step, batch in enumerate(data_loader):
            with accelerator.accumulate(if_I.model):
                bsz = batch['input_ids'].size(0)
                embeds = t5.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                ).last_hidden_state
                if_I_kwargs['text_emb'] = embeds
                timesteps = torch.randint(low=1, high=num_timesteps-1, size=(bsz,)).to(if_I.device)
                loss = diffusion.training_losses(
                    if_I.model, 
                    batch['pixel_values'],
                    timesteps,
                    if_I_kwargs
                )['loss'].sum() # TODO: Add prior_loss_weight
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = batch['input_ids'].new_ones((len(t5.tokenizer),), dtype=torch.bool)
                index_no_updates[min(vtoken_ids) : max(vtoken_ids) + 1] = False

                with torch.no_grad():
                    accelerator.unwrap_model(t5.model).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = out_dir / f"unet-step-{global_step}.bin"
                    save_ckpt(accelerator, if_I.model, t5.model, save_path, vtoken_ids)
                    logger.info(f"Saved state to {save_path}")
                if global_step % args.log_steps == 0:
                    logger.info(f"[Step {global_step}] Loss: {loss.detach().item()}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    logger.info("Finished training")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    define_args(parser)
    define_adapter_args(parser)
    args = parser.parse_args()
    main(args)
