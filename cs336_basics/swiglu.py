from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from cs336_basics.linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dtype, device = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        
        if d_ff is None:
            # Wrong:
            # a = d_model*8/3
            # b = (a + 63) // 64
            # d_ff = b*64
            #
            # Reason:
            # The original version computes through floats, so d_ff can become a
            # floating-point value. Layer sizes must be integers. We round up to
            # the next multiple of 64 explicitly and cast to int.
            d_ff = int(64 * math.ceil((8 * d_model / 3) / 64))
        self.d_ff = d_ff
        
        # Wrong:
        # self.Linear1 = Linear(d_model, d_ff, device, dtype)
        #
        # Reason:
        # The assignment/test weights are conventionally named w1/w2/w3. Keeping
        # those names makes state-dict loading straightforward.
        # Wrong:
        # self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
        # Wrong:
        # self.Linear2 = Linear(d_ff, d_model, device, dtype)
        # Wrong:
        # self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        
        # Wrong:
        # self.Linear3 = Linear(d_model, d_ff, device, dtype)
        # Wrong:
        # self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

        # Preserve your original attribute names as aliases so the rest of your
        # code can keep working with minimal change.
        self.Linear1 = self.w1
        self.Linear2 = self.w2
        self.Linear3 = self.w3
    
    def silu_x(self, x):
        # Equivalent to x * sigmoid(x), but using the built-in op is clearer and
        # matches PyTorch numerics directly.
        return F.silu(x)
        
    def forward(self, x):
        x1 = self.Linear1(x)
        x1 = self.silu_x(x1)
        x3 = self.Linear3(x)
        x = x1*x3
        
        return self.Linear2(x)
    
    
        
    
        
        
        
        
