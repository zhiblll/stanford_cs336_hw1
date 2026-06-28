import torch
from torch import nn



class RotaryPositionalEmbedding(nn.Module):
    
    def __init__(self, max_seq_length, dk, theta, device = None):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.dk = dk
        self.theta = theta
        self.device = device
        
        pair_idx = torch.arange(end = dk // 2, device = self.device)
        
        inv_freq = theta ** (-2*pair_idx/self.dk)
        
        position = torch.arange(self.max_seq_length, device = self.device)
        
        angles = position[: , None]*inv_freq[None, :]
        
        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)
        
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        
    def rotate_pairs(self, x: torch.tensor):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        x_stack = torch.stack((-x_odd, x_even), dim = -1)
        
        return torch.flatten(x_stack, -2)
    
    def forward(self, x, token_pos):
        cos = self.cos_cached
        sin = self.sin_cached
        
        cos = cos.repeat_interleave(2, dim=-1).to(dtype=x.dtype, device=x.device)
        sin = sin.repeat_interleave(2, dim=-1).to(dtype=x.dtype, device=x.device)
        
        x_rotate = self.rotate_pairs(x)
        sin = sin[token_pos]
        cos = cos[token_pos]
        if cos.ndim == x.ndim - 1:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        
        return x_rotate*sin + x*cos
        
        
        
        
    
        
        