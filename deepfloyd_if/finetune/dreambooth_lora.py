import argparse, logging, sys

from deepfloyd_if.modules import T5Embedder, IFStageI
from deepfloyd_if.finetune.dreambooth import define_args
from deepfloyd_if.finetune.lora import inject_trainable_lora


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)

logger = logging.getLogger(__name__)


def main(args) -> None:
    if_I = IFStageI("IF-I-XL-v1.0", device="cuda", model_kwargs={"precision": 32, "use_checkpoint": True})
    if_I.model.to(dtype=if_I.model.dtype)

    lora_params, module_names = inject_trainable_lora(if_I.model, verbose=True)
    print(lora_params, len(lora_params))
    print(module_names)
    print(if_I.model)



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    args = arg_parser.parse_args()
    main(args)
