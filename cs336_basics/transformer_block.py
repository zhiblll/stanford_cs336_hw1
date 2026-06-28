import torch
from torch import nn
from cs336_basics import multihead_att
from cs336_basics import swiglu
from cs336_basics import rms_norm
class transformerBlock(nn.Module):
    def __init__(self, d_model, dff, num_head, theta, max_seq_len, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.num_head = num_head
        self.device = device
        self.dtype = dtype
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        self.att = multihead_att.Multi_head_att(self.d_model, self.num_head, self.theta, self.max_seq_len)
        self.ff = swiglu.SwiGLU(self.d_model, self.dff, self.dtype, self.device)
        self.rms_norm1 = rms_norm.RMSNorm(self.d_model, device = self.device, dtype = self.dtype)
        self.rms_norm2 = rms_norm.RMSNorm(self.d_model, device = self.device, dtype = self.dtype)
        
    def forward(self, x):
        x_norm_1 = self.rms_norm1(x)
        x1 = self.att(x_norm_1)
        
        x = x + x1
        
        x_norm_2 = self.rms_norm2(x)
        x2 = self.ff(x_norm_2)
        
        x = x + x2
        
        return x
    
class TransformerBlockPostNorm(nn.Module):
    def __init__(self, d_model, dff, num_head, theta, max_seq_len, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.num_head = num_head
        self.device = device
        self.dtype = dtype
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        self.att = multihead_att.Multi_head_att(self.d_model, self.num_head, self.theta, self.max_seq_len)
        self.ff = swiglu.SwiGLU(self.d_model, self.dff, self.dtype, self.device)
        self.rms_norm1 = rms_norm.RMSNorm(self.d_model, device = self.device, dtype = self.dtype)
        self.rms_norm2 = rms_norm.RMSNorm(self.d_model, device = self.device, dtype = self.dtype)
        
    def forward(self, x):
        x = self.rms_norm1(x + self.att(x))
        x = self.rms_norm2(x + self.ff(x))

        return x 

class TransformerBlockNoROPE(nn.Module):
    def __init__(self, d_model, dff, num_head, theta, max_seq_len, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.num_head = num_head
        self.device = device
        self.dtype = dtype
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        self.att = multihead_att.Multi_head_att_no_rope(self.d_model, self.num_head, self.theta, self.max_seq_len)
        self.ff = swiglu.SwiGLU(self.d_model, self.dff, self.dtype, self.device)
        self.rms_norm1 = rms_norm.RMSNorm(self.d_model, device = self.device, dtype = self.dtype)
        self.rms_norm2 = rms_norm.RMSNorm(self.d_model, device = self.device, dtype = self.dtype)
        
    def forward(self, x):
        x_norm_1 = self.rms_norm1(x)
        x1 = self.att(x_norm_1)
        
        x = x + x1
        
        x_norm_2 = self.rms_norm2(x)
        x2 = self.ff(x_norm_2)
        
        x = x + x2
        
        return x