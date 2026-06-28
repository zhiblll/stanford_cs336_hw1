import torch
import math
from torch import nn

def softmax(x: torch.tensor, dim: int) -> torch.tensor:
    max_value = torch.max(x, dim = dim, keepdim = True).values
    
    x_norm = x - max_value
    
    x_exp = torch.exp(x_norm)
    
    x_sum = torch.sum(x_exp, dim = dim, keepdim = True)
    
    return x_exp/x_sum
    
    
def scaled_dot_product_attention(q, k, v, mask = None
) -> torch.Tensor:
    
    dk = q.shape[-1]
    k_T = torch.transpose(k, dim0 = -2, dim1 = -1)
    a = q@k_T
    b = a/math.sqrt(dk)
    if mask is not None:
        b = b.masked_fill(~mask, float("-inf"))
    
    c = softmax(b, dim = -1)

    
    d = c@v
    
    return d