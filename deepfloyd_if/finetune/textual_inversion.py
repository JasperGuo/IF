from typing import List, Dict, Tuple
import argparse, os, random, math, logging, sys

import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from transformers import AutoTokenizer
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from diffusers import get_scheduler

from utils import freeze_params
from deepfloyd_if.modules import T5Embedder, IFStageI

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = get_logger(__name__, log_level='INFO')

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

object_variations = [
    "figure, white background",
    "woman, white background",
    "doll, white background",
    "puppet, white background"
]

style_variations = []

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class TextualInversionDataset(Dataset):

    def __init__(
        self,
        data_root: str,
        vtoken_seq: str,
        tokenizer: AutoTokenizer,
        repeats=1,
        size: int = 64,
        flip_p: float =0.5,
        learnable_property: str = "style",
        use_variations: bool = False
    ) -> None:
        super().__init__()

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.vtoken_seq = vtoken_seq

        # data
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.num_images = len(self.image_paths)
        self._length = self.num_images * repeats

        # Prompt
        self.learnable_property = learnable_property
        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.variations = style_variations if learnable_property == "style" else object_variations
        self.use_variations = use_variations

        # image
        self.size = size
        self.flip_p = flip_p
        self.transforms = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=self.flip_p),
            T.ToTensor()
        ])
        self.flip_transform = T.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Dict:
        example = {}
        image = Image.open(self.image_paths[index % self.num_images])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self.use_variations:
            _vtoken_seq = f"{self.vtoken_seq} {random.choice(self.variations)}" 
            prompt = random.choice(self.templates).format(_vtoken_seq)
        else:
            prompt = random.choice(self.templates).format(self.vtoken_seq)
        prompt_tokens_and_mask = self.tokenizer(
            prompt,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        example['input_ids'] = prompt_tokens_and_mask['input_ids'][0]
        example['attention_mask'] = prompt_tokens_and_mask['attention_mask'][0]

        pixel_values = self.transforms(image)
        pixel_values = pixel_values * 2.0 - 1.0
        example['pixel_values'] = pixel_values
        return example


def add_vtokens(t5: T5Embedder, num_tokens: int = 1, prefix: str = "sks", init_token: str = None) -> Tuple[str, List[int]]:
    tokens_to_add = [prefix] if num_tokens == 1 else [f"{prefix}_{i}" for i in range(num_tokens)]
    num_added_tokens = t5.tokenizer.add_tokens(tokens_to_add)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {tokens_to_add}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    vtoken_ids = t5.tokenizer.convert_tokens_to_ids(tokens_to_add)
    t5.model.resize_token_embeddings(len(t5.tokenizer))
    if init_token is not None:
        init_token_ids = t5.tokenizer.encode(init_token, add_special_tokens=False)
        if len(init_token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")
        init_token_id = init_token_ids[0]
        print(f"Init token: {init_token}, token id: {init_token_id}")
        # init
        token_embeds = t5.model.get_input_embeddings().weight.data
        with torch.no_grad():
            for vid in vtoken_ids:
                token_embeds[vid] = token_embeds[init_token_id].clone()
    vtoken_seq = " ".join(tokens_to_add)
    print(f"VToken ids: {vtoken_ids}")
    return vtoken_seq, vtoken_ids


def force_training_grad(model: nn.Module, do_train: bool = True, do_grad: bool = True) -> None:
    model.training = do_train
    model.requires_grad_ = do_grad
    for module in model.children():
        force_training_grad(module)


def save_progress(text_encoder, vtoken_ids: List[int], accelerator: Accelerator, save_path: str):
    logger.info("Saving embeddings")
    learned_embeds_dict = dict()
    embeddings = accelerator.unwrap_model(text_encoder).get_input_embeddings()
    for vid in vtoken_ids:
        learned_embeds = embeddings.weight[vid]
        learned_embeds_dict[vid] = learned_embeds.detach().cpu()
    torch.save(learned_embeds_dict, save_path)


def main(args) -> None:
    set_seed(args.seed)

    t5 = T5Embedder(device=args.device, torch_dtype=torch.float32, use_offload_folder=None)
    vtoken_seq, vtoken_ids = add_vtokens(t5, num_tokens=args.num_vtokens, init_token=args.init_token)

    dataset = TextualInversionDataset(
        data_root=args.data_root,
        vtoken_seq=vtoken_seq,
        tokenizer=t5.tokenizer,
        repeats=100,
        learnable_property=args.learnable_property,
        use_variations=args.use_variations
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # initialize stage I model & freeze its parameters
    if_I = IFStageI(args.if_I, device=args.device, model_kwargs={"precision": 32})
    if_I.model.to(dtype=if_I.model.dtype).eval()
    if_I.model.requires_grad_(False)

    # initialize t5 & freeze its encoder parameters
    t5.model = t5.model.train()
    freeze_params(t5.model.encoder.parameters())
    for param in t5.model.get_input_embeddings().parameters():
        param.requires_grad = True

    # optimizers
    lr = args.lr
    if args.scale_lr:
        lr = lr * args.batch_size * args.gradient_accumulation_steps
    optimizer = torch.optim.AdamW(
        t5.model.get_input_embeddings().parameters(),  # only optimize the embeddings
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
        mixed_precision="no",
        log_with=args.report_to,
    )
    text_encoder, optimizer, data_loader, lr_scheduler = accelerator.prepare(
        t5.model, optimizer, data_loader, lr_scheduler
    )

    accelerator.init_trackers("textual_inversion", config=vars(args))

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
    scale = torch.tensor(num_timesteps, dtype=torch.float).to(if_I.device)
    if_I_kwargs = {
        'guidance_scale': 7.0
    }

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    for epoch in range(num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(data_loader):
            with accelerator.accumulate(text_encoder):
                bsz = batch['input_ids'].size(0)
                embeds = text_encoder(
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
                )['loss']
                accelerator.backward(loss)

                # grad
                grads = text_encoder.get_input_embeddings().weight.grad.clone().detach()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                 # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = batch['input_ids'].new_ones((len(t5.tokenizer),), dtype=torch.bool)
                index_no_updates[min(vtoken_ids) : max(vtoken_ids) + 1] = False

                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"learned_embeds-step-{global_step}.bin")
                    save_progress(text_encoder, vtoken_ids, accelerator, save_path)
                if global_step % args.log_steps == 0:
                    logger.info(f"[Step {global_step}] Loss: {loss.detach().item()}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "grad": torch.abs(grads).sum().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    logger.info("Finished training")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-vtokens", type=int, default=1, help="number of vtokens")
    parser.add_argument("--init-token", type=str, required=True, help="init token")
    parser.add_argument("--data-root", type=str, required=True, help="data root")
    parser.add_argument("--batch-size", type=int, required=True, help="batch size for training")
    parser.add_argument("--learnable-property", type=str, default="object", choices=["object", "style"])
    parser.add_argument("--device", type=str, default="cuda:0", help="device to train")
    parser.add_argument("--if-I", type=str, default="IF-II-M-v1.0", help="IF-I model name")

    # training hyper-parameters
    parser.add_argument("--lr", type=float, default=5e-04, help="learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--scale_lr", action='store_true', help="scale learning rate with batch size, gradient accumulation step, and num_gpu")
    parser.add_argument("--max_train_steps", type=int, default=2000, help="number of training steps")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--save_steps", type=int, default=500, help="number of steps to save embeddings")
    parser.add_argument("--output_dir", type=str, required=True, help="output dir to store learned embeddings")
    parser.add_argument("--log_steps", type=int, default=50, help="number of steps to log graident & loss")
    parser.add_argument("--use_variations", action='store_true', help="use object variations")

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

    args = parser.parse_args()
    main(args)
