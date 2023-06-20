from time import time
from pathlib import Path
from typing import Dict, List
import argparse, logging, sys, math, itertools

from tqdm import tqdm
from PIL import Image
from PIL.ImageOps import exif_transpose
from accelerate.utils import set_seed
from accelerate import Accelerator
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torchvision import transforms as T
from diffusers import get_scheduler
import bitsandbytes as bnb

from deepfloyd_if.pipelines import dream
from deepfloyd_if.modules import T5Embedder, IFStageI
from deepfloyd_if.finetune.utils import freeze_params, clean_cuda_cache
from deepfloyd_if.finetune.lora import inject_trainable_lora, save_lora_weight


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)

logger = logging.getLogger(__name__)


def define_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--lora", action='store_true', help="whether to train with lora")
    parser.add_argument("--lora_r", type=int, default=4, help="lora rank")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="lora scale when merged with weights")

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


class DreamBoothDataset(Dataset):

    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        instance_prompt_encoder_states: torch.Tensor,
        tokenizer: AutoTokenizer,
        class_data_root: str,
        class_prompt: str,
        class_prompt_encoder_states: torch.Tensor,
        size: int = 64,
        flip_p: float = 0.0 # Do not enable horizontal flip by default
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer

        # instance
        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self.instance_prompt_encoder_states = instance_prompt_encoder_states

        # Prior preservation
        self.class_images_path = list(Path(class_data_root).iterdir())
        self.num_class_images = len(self.class_images_path)
        self.class_prompt = class_prompt
        self.class_prompt_encoder_states = class_prompt_encoder_states

        self.size = size
        self.flip_p = flip_p
        self.transforms = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=self.flip_p),
            T.ToTensor()
        ])
        self.flip_transform = T.RandomHorizontalFlip(p=self.flip_p)

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
        example["instance_prompt_embeds"] = self.instance_prompt_encoder_states

        # prior image
        class_image = Image.open(self.class_images_path[index % self.num_class_images])
        class_image = exif_transpose(class_image)

        if not class_image.mode == "RGB":
            class_image = class_image.convert("RGB")
        example["class_image"] = self.transforms(class_image)
        example["class_prompt_embeds"] = self.class_prompt_encoder_states

        return example


def collate_fn(examples: List[Dict]) -> Dict:
    pixel_values = [ex["instance_image"] for ex in examples]
    pixel_values.extend([ex['class_image'] for ex in examples])
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    embeds = [ex['instance_prompt_embeds'] for ex in examples]
    embeds.extend([ex['class_prompt_embeds'] for ex in examples])
    embeds = torch.stack(embeds)

    batch = {
        "embeds": embeds,
        "pixel_values": pixel_values,
    }
    return batch


def prepare_class_images(
    class_data_root: str,
    num_class_images: int, 
    class_prompt: str,
    t5: T5Embedder, 
    if_I: IFStageI,
) -> None:
    root = Path(class_data_root)
    root.mkdir(exist_ok=True, parents=True)
    diff = num_class_images - len(list(root.iterdir()))
    if diff <= 0:
        return

    # inference
    with torch.inference_mode():
        for _ in tqdm(range(diff)):
            result = dream(
                t5=t5, if_I=if_I,
                prompt=class_prompt,
                if_I_kwargs={
                    "guidance_scale": 7.0,
                    "sample_timestep_respacing": "smart100",
                },
            )
            result['I'][0].save(root / f"class_prior_{int(time())}.png")

    # Clean memory
    clean_cuda_cache()


def get_text_encoder_states(t5: T5Embedder, prompt: str) -> torch.Tensor:
    prompt_tokens_and_mask = t5.tokenizer(
        prompt,
        max_length=77,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    with torch.inference_mode():
        embeds = t5.model(
            input_ids=prompt_tokens_and_mask['input_ids'],
            attention_mask=prompt_tokens_and_mask['attention_mask']
        ).last_hidden_state.squeeze(dim=0)
    return embeds


def save_ckpt(accelerator: Accelerator, model, save_path: str, use_lora: bool = False) -> None:
    unet = accelerator.unwrap_model(model)
    if not use_lora:
        torch.save(unet, save_path)
    else:
        save_lora_weight(unet, save_path, dtype=torch.float32)


def main(args) -> None:
    set_seed(args.seed)
    logger.info(f"Instance prompt: {args.instance_prompt}")
    logger.info(f"Class prompt: {args.class_prompt}")

    t5 = T5Embedder(device="cpu", torch_dtype=torch.float32, use_offload_folder=None)
    freeze_params(t5.model.parameters())

    if_I = IFStageI(
        args.if_I, 
        device=args.device, 
        model_kwargs={
            "precision": 32, 
            "use_checkpoint": args.use_gradient_checkpoint
        }
    )
    if_I.model.to(dtype=if_I.model.dtype)

    prepare_class_images(
        class_data_root=args.class_data_root,
        num_class_images=args.num_class_images,
        class_prompt=args.class_prompt,
        t5=t5, if_I=if_I
    )

    # prepare dataset
    instance_prompt_encoder_states = get_text_encoder_states(t5, args.instance_prompt)
    class_prompt_encoder_states = get_text_encoder_states(t5, args.class_prompt)

    dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_root,
        instance_prompt=args.instance_prompt,
        instance_prompt_encoder_states=instance_prompt_encoder_states,
        tokenizer=t5.tokenizer,
        class_data_root=args.class_data_root,
        class_prompt=args.class_prompt,
        class_prompt_encoder_states=class_prompt_encoder_states,
    )
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if_I.model.train()
    # optimizers
    lr = args.lr
    if args.scale_lr:
        lr = lr * args.batch_size * args.gradient_accumulation_steps
    #  torch.optim.AdamW. Here, use 8bit Adam
    adamw_opt = bnb.optim.Adam8bit if args.use_8bitadam else torch.optim.AdamW
    if args.lora:
        logger.info(f"Trained with LoRA(r={args.lora_r}, scale={args.lora_scale})")
        if_I.model.unet.requires_grad_(False)
        lora_trainable_parameters, lora_module_names = inject_trainable_lora(
            if_I.model,
            verbose=True,
            r=args.lora_r,
            scale=args.lora_scale
        )
        optimizer = adamw_opt(
            itertools.chain(*lora_trainable_parameters), # only optimize the embeddings
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08
        )
    else:
        optimizer = adamw_opt(
            if_I.model.parameters(),  # only optimize the embeddings
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
    accelerator.init_trackers("dreambooth", config=vars(args))
    if_I.model, optimizer, data_loader, lr_scheduler = accelerator.prepare(
        if_I.model, optimizer, data_loader, lr_scheduler
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

    for epoch in range(num_train_epochs):
        if_I.model.train()
        for step, batch in enumerate(data_loader):
            with accelerator.accumulate(if_I.model):
                bsz = batch['embeds'].size(0)
                timesteps = torch.randint(low=1, high=num_timesteps-1, size=(bsz,)).to(if_I.device)
                if_I_kwargs['text_emb'] = batch['embeds']
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

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = out_dir / f"unet-step-{global_step}.bin"
                    save_ckpt(accelerator, if_I.model, save_path, use_lora=args.lora)
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
