from __future__ import annotations

import torch
from torch import nn
import math

class Linear(nn.Module):
    def __init__ (self, in_features: int, out_features: int, weight: torch.Tensor | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None,) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if weight is not None:
            self.weight = weight
        else:
            self.weight: nn.Parameter =nn.Parameter(torch.empty(
                out_features, in_features, 
                device = device, 
                dtype = dtype))
            self.reset_parameters()

        
    def reset_parameters(self) -> None:
        # TODO:
        # 1) compute sigma = sqrt(2 / (in_features + out_features))
        # 2) initialize self.W with trunc_normal_
        # 3) truncate to [-3*sigma, 3*sigma]
        sigma =  math.sqrt(2 / (self.in_features + self.out_features))
        # nn.init.trunc_normal_(tensor = self.W std = sigma, a=-3.0 * sigma, b=3.0 * sigma )
        nn.init.trunc_normal_(
        self.weight,
        mean=0.0,
        std=sigma,
        a=-3.0 * sigma,
        b=3.0 * sigma,
    )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO:
        # Apply linear transform on the final dimension of x
        # Input shape: (..., in_features)
        # Output shape: (..., out_features)
        
        return x@self.weight.transpose(0,1)