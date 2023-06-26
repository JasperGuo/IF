from pathlib import Path
from typing import Dict, List, Tuple
import argparse, logging, sys, itertools, math

from tqdm import tqdm
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torchvision import transforms as T
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import get_scheduler
import bitsandbytes as bnb

from deepfloyd_if.modules import T5Embedder, IFStageI
from deepfloyd_if.finetune.utils import freeze_params
from deepfloyd_if.finetune.textual_inversion import add_vtokens
from deepfloyd_if.finetune.dreambooth import prepare_class_images
from deepfloyd_if.finetune.lora import _find_modules, UNET_DEFAULT_TARGET_REPLACE


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)

logger = logging.getLogger(__name__)


def define_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-vtokens", type=int, default=1, help="number of vtokens")
    parser.add_argument("--init-token", type=str, required=True, help="init token")

    parser.add_argument("--instance-prompt", type=str, required=True, help="instance prompt")
    parser.add_argument("--instance-data-root", type=str, required=True, help="instance data root")
    parser.add_argument("--class-data-root", type=str, required=True, help="class data root")
    parser.add_argument("--class-prompt", type=str, required=True, help="class prompt")
    parser.add_argument("--num-class-images", type=int, required=True, help="num class images to synthesis", default=1000)

    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--batch-size", type=int, required=True, help="batch size for training")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to train")
    parser.add_argument("--if-I", type=str, default="IF-II-M-v1.0", help="IF-I model name")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir to store learned embeddings")

    # training hyper-parameters
    parser.add_argument("--lr", type=float, default=5e-04, help="learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--scale_lr", action='store_true', help="scale learning rate with batch size, gradient accumulation step, and num_gpu")
    parser.add_argument("--max_train_steps", type=int, default=2000, help="number of training steps")
    parser.add_argument("--save_steps", type=int, default=500, help="number of steps to save embeddings")
    parser.add_argument("--log_steps", type=int, default=50, help="number of steps to log graident & loss")
    parser.add_argument("--use_gradient_checkpoint", action='store_true', help="use gradient checkpoint")
    parser.add_argument("--use_8bitadam", action='store_true', help="whether to use 8bit adam (save memory)")

    # learning rate schduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )

    # acceleartor
    parser.add_argument("--report_to", type=str, default="wandb", help="experiment tracker")


class CustomDiffusionDataset(Dataset):

    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        tokenizer: AutoTokenizer,
        class_data_root: str,
        class_prompt: str,
        size: int = 64,
        flip_p: float = 0.0 # Do not enable horizontal flip by default
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer

        # instance
        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt

        # Prior preservation
        self.class_images_path = list(Path(class_data_root).iterdir())
        self.num_class_images = len(self.class_images_path)
        self.class_prompt = class_prompt

        self.size = size
        self.flip_p = flip_p
        self.transforms = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=self.flip_p),
            T.ToTensor()
        ])

    def __len__(self):
        return max(self.num_class_images, self.num_instance_images)

    def __getitem__(self, index: int) -> Dict:
        example = dict()

        # instance image
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_image"] = self.transforms(instance_image)
        instance_prompt_tokens_and_mask = self.tokenizer(
            self.instance_prompt,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        example['instance_input_ids'] = instance_prompt_tokens_and_mask['input_ids'][0]
        example['instance_attention_mask'] = instance_prompt_tokens_and_mask['attention_mask'][0]

        # prior image
        class_image = Image.open(self.class_images_path[index % self.num_class_images])
        class_image = exif_transpose(class_image)
        if not class_image.mode == "RGB":
            class_image = class_image.convert("RGB")
        example["class_image"] = self.transforms(class_image)

        class_prompt_tokens_and_mask = self.tokenizer(
            self.class_prompt,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        example['class_input_ids'] = class_prompt_tokens_and_mask['input_ids'][0]
        example['class_attention_mask'] = class_prompt_tokens_and_mask['attention_mask'][0]

        return example


def collate_fn(examples: List[Dict]) -> Dict:
    pixel_values = [ex["instance_image"] for ex in examples]
    pixel_values.extend([ex['class_image'] for ex in examples])
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = [ex['instance_input_ids'] for ex in examples]
    input_ids.extend([ex['class_input_ids'] for ex in examples])
    input_ids = torch.stack(input_ids)

    attention_mask = [ex['instance_attention_mask'] for ex in examples]
    attention_mask.extend([ex['class_attention_mask'] for ex in examples])
    attention_mask = torch.stack(attention_mask)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values
    }
    return batch


def find_crossattn_kv_parameters(model: nn.Module) -> Tuple[List, List[str]]:
    require_grad_params = []
    names = []

    for _module, name, _child_module in _find_modules(
        model, UNET_DEFAULT_TARGET_REPLACE, search_class=[nn.Conv1d]
    ):
        if name != "encoder_kv":
            continue
        for param in _module._modules[name].parameters():
            param.requires_grad = True
        require_grad_params.append(_module._modules[name].parameters())
        names.append(name)
    return require_grad_params, names


def save_ckpt(accelerator: Accelerator, model, text_encoder, save_path: str, vtoken_ids: List[int]) -> None:
    logger.info(f"Saving unet: {save_path}")
    unet = accelerator.unwrap_model(model)
    torch.save(unet, save_path)

    logger.info(f"Saving vtoken embeddings: {save_path.replace('.bin', '-embeds.bin')}")
    learned_embeds_dict = dict()
    embeddings = accelerator.unwrap_model(text_encoder).get_input_embeddings()
    for vid in vtoken_ids:
        learned_embeds = embeddings.weight[vid]
        learned_embeds_dict[vid] = learned_embeds.detach().cpu()
    torch.save(learned_embeds_dict, save_path.replace(".bin", "-embeds.bin"))


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
    if_I_trainable_parameters, if_I_module_names = find_crossattn_kv_parameters(if_I.model)
    logger.info(if_I_module_names)
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
    accelerator.init_trackers("custom_diffusion", config=vars(args))
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
    args = parser.parse_args()
    main(args)
