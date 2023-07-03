import math, re
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfloyd_if.model.nn import get_activation
from deepfloyd_if.model.unet import AttentionBlock, UNetModel, AttentionAdapterVariant


def _copy_conv_weights(conv: nn.Module, ref_conv: nn.Module) -> None:
    conv.weight = ref_conv.weight
    if ref_conv.bias is not None:
        conv.bias = ref_conv.bias


def _copy_norm_weights(norm: nn.Module, ref_norm: nn.Module) -> None:
    if ref_norm.weight is not None:
        norm.weight = ref_norm.weight
    if ref_norm.bias is not None:
        norm.bias = ref_norm.bias


class BottleneckAdapter(nn.Module):

    def __init__(self, channels: int, bottleneck_channels: int, dtype: str, ref_adapter: "BottleneckAdapter" = None) -> None:
        super().__init__()
        self.dtype = dtype
        self.channels = channels
        self.bottleneck_channels = bottleneck_channels

        self.activation = get_activation("gelu")
        if ref_adapter is None:
            self.down_weight = nn.Parameter(torch.zeros(self.bottleneck_channels, self.channels, 1, dtype=self.dtype), requires_grad=True)
            self.down_bias = nn.Parameter(torch.zeros(self.bottleneck_channels, dtype=self.dtype), requires_grad=True)
            self.up_weight = nn.Parameter(torch.zeros(self.channels, self.bottleneck_channels, 1, dtype=self.dtype), requires_grad=True)
            self.up_bias = nn.Parameter(torch.zeros(self.channels, dtype=self.dtype), requires_grad=True)
            nn.init.kaiming_normal_(self.down_weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_bias)
            nn.init.zeros_(self.up_weight)
            nn.init.zeros_(self.up_bias)
        else:
            # copy the weights & bias
            self.down_weight = ref_adapter.down_weight
            self.down_bias = ref_adapter.down_bias
            self.up_weight = ref_adapter.up_weight
            self.up_bias = ref_adapter.up_bias

    def forward(self, x):
        h = F.conv1d(x, self.down_weight, bias=self.down_bias, stride=1)
        h = self.activation(h)
        h = F.conv1d(h,self.up_weight, self.up_bias, stride=1)
        return h


class AdapterAttentionBlock(nn.Module, AttentionAdapterVariant):

    def __init__(
        self,
        attn_block: AttentionBlock,
        adapter_block: BottleneckAdapter,
        adapter_scale: float = 1.0
    ) -> None:
        super().__init__()
        self.attn_block = attn_block
        self.adapter_block = adapter_block
        self.adapter_scale = adapter_scale

    def forward(self, x, encoder_out=None):
        h = self.attn_block(x, encoder_out)
        b, c, *spatial = h.shape
        y = self.adapter_block(h.view(b, c, -1))
        return y.reshape(b, c, *spatial) * self.adapter_scale + h


def inject_adapter(model: UNetModel, bottlenect_r: int = 2, adapter_scale: float = 1.0) -> List:
    require_grad_params = []

    adapter_attn_blocks = list()
    for fullname, module in model.named_modules():
        if module.__class__.__name__ == "AttentionBlock":
            # replace it
            adapter = BottleneckAdapter(
                channels=module.channels,
                bottleneck_channels=module.channels // bottlenect_r,
                dtype=module.dtype,
                ref_adapter=None
            )
            adapter_attn_block = AdapterAttentionBlock(
                attn_block=module,
                adapter_block=adapter,
                adapter_scale=adapter_scale
            )
            # replace the module
            require_grad_params.append(adapter.parameters())
            adapter_attn_blocks.append((fullname, adapter_attn_block,))
    for fn, aab in adapter_attn_blocks:
        m = model
        paths = fn.split(".")
        for pid, name in enumerate(paths):
            if pid == len(paths) - 1:
                if not re.match(r'^\d+$', name):
                    setattr(m, name, aab)
                else:
                    m[int(name)] = aab
            else:
                if not re.match(r'^\d+$', name):
                    m = getattr(m, name)
                else:
                    m = m[int(name)]
    return require_grad_params
