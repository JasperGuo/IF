from typing import List

import torch


def freeze_params(params: List) -> None:
    for param in params:
        param.requires_grad = False


def clean_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
