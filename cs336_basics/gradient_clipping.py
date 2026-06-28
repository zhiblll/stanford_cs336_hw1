import torch
import math
def gradient_clipping(params, max_value, eps = 1e-6):
    
    total_sum = 0
    
    for p in params:
        
        if p.grad is None:
            continue
        
        total_sum += (p.grad ** 2).sum().item()
        
    
    total_sum = math.sqrt(total_sum)
    
    if total_sum > max_value:
        scale = max_value/(total_sum + eps)
        
        for p in params:
           if p.grad is None:
                continue
           p.grad = scale * p.grad
        