import torch
from torch import nn
import math
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.attention import scaled_dot_product_attention
class Multi_head_att(nn.Module):
    def __init__(self, d_model, num_head, theta, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        
        self.head_dim = d_model // num_head
        self.rope = RotaryPositionalEmbedding(max_seq_length= max_seq_len, dk = self.head_dim, theta=theta)
        self.wq = Linear(d_model, d_model)
        self.wk = Linear(d_model, d_model)
        self.wv = Linear(d_model, d_model)
        self.w_out = Linear(d_model, d_model)
    
    def forward(self, x):
        batch_num, seq_len = x.shape[0:2]
        q_proj = self.wq(x)
        k_proj = self.wk(x)
        v_proj = self.wv(x)
        
        q_proj = q_proj.reshape(batch_num, seq_len, self.num_head, self.head_dim)
        k_proj = k_proj.reshape(batch_num, seq_len,self.num_head, self.head_dim)
        v_proj = v_proj.reshape(batch_num, seq_len,self.num_head, self.head_dim)
        q_proj = torch.transpose(q_proj, 1, 2)
        k_proj = torch.transpose(k_proj, 1, 2)
        v_proj = torch.transpose(v_proj, 1, 2)
        
        token_positions = torch.arange(0, seq_len, device = x.device)
        # token_positions = token_positions.view(1, 1, seq_len).expand(batch_num, seq_len, seq_len)
        q_rope = self.rope(q_proj, token_positions)
        k_rope = self.rope(k_proj, token_positions)
        
        # out = scaled_dot_product_attention(q_rope, k_rope, v_proj)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        out = scaled_dot_product_attention(q_rope, k_rope, v_proj, mask)
        
        
        out = torch.transpose(out, 1, 2)
        out = out.reshape(batch_num, seq_len, self.d_model)
        
        return self.w_out(out)

class Multi_head_att_no_rope(nn.Module):
    def __init__(self, d_model, num_head, theta, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        
        self.head_dim = d_model // num_head
        self.wq = Linear(d_model, d_model)
        self.wk = Linear(d_model, d_model)
        self.wv = Linear(d_model, d_model)
        self.w_out = Linear(d_model, d_model)
    
    def forward(self, x):
        batch_num, seq_len = x.shape[0:2]
        q_proj = self.wq(x)
        k_proj = self.wk(x)
        v_proj = self.wv(x)
        
        q_proj = q_proj.reshape(batch_num, seq_len, self.num_head, self.head_dim)
        k_proj = k_proj.reshape(batch_num, seq_len,self.num_head, self.head_dim)
        v_proj = v_proj.reshape(batch_num, seq_len,self.num_head, self.head_dim)
        q_proj = torch.transpose(q_proj, 1, 2)
        k_proj = torch.transpose(k_proj, 1, 2)
        v_proj = torch.transpose(v_proj, 1, 2)
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        out = scaled_dot_product_attention(q_proj, k_proj, v_proj, mask)
        
        
        out = torch.transpose(out, 1, 2)
        out = out.reshape(batch_num, seq_len, self.d_model)
        
        return self.w_out(out)
        
        