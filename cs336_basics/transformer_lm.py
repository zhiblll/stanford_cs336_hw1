import torch
from torch import nn
from cs336_basics.Embedding import Embedding
from cs336_basics import transformer_block
from cs336_basics import rms_norm
from cs336_basics import linear
class transformerlm (nn.Module):
    def __init__(self, num_layers, vocab_size, max_seq_length, d_ff, d_model, num_heads, theta, device, dtype):
        
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = max_seq_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.device = device
        self.dtype = dtype
        
        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        
        self.layers = nn.ModuleList([transformer_block.transformerBlock(self.d_model, self.d_ff, self.num_heads, self.theta, self.context_length, self.device, self.dtype) for _ in range(num_layers)])
        
        self.final_norm = rms_norm.RMSNorm(self.d_model, device=self.device, dtype=self.dtype)
        
        self.lm_head = linear.Linear(
            in_features=d_model,
            out_features=vocab_size,
            device= self.device,
            dtype=self.dtype,
        )

    def forward(self, x):
        
        x = self.token_embedding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x_norm = self.final_norm(x)
        
        logist = self.lm_head(x_norm)
        
        return logist

class posttransformerlm(nn.Module):
    def __init__(self, num_layers, vocab_size, max_seq_length, d_ff, d_model, num_heads, theta, device, dtype):
        
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = max_seq_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.device = device
        self.dtype = dtype
        
        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        
        self.layers = nn.ModuleList([transformer_block.TransformerBlockPostNorm(self.d_model, self.d_ff, self.num_heads, self.theta, self.context_length, self.device, self.dtype) for _ in range(num_layers)])
        
        self.final_norm = rms_norm.RMSNorm(self.d_model, device=self.device, dtype=self.dtype)
        
        self.lm_head = linear.Linear(
            in_features=d_model,
            out_features=vocab_size,
            device= self.device,
            dtype=self.dtype,
        )

    def forward(self, x):
        
        x = self.token_embedding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x_norm = self.final_norm(x)
        
        logist = self.lm_head(x_norm)
        
        return logist
class transformerlm_no_rope (nn.Module):
    def __init__(self, num_layers, vocab_size, max_seq_length, d_ff, d_model, num_heads, theta, device, dtype):
        
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = max_seq_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.device = device
        self.dtype = dtype
        
        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        
        self.layers = nn.ModuleList([transformer_block.TransformerBlockNoROPE(self.d_model, self.d_ff, self.num_heads, self.theta, self.context_length, self.device, self.dtype) for _ in range(num_layers)])
        
        self.final_norm = rms_norm.RMSNorm(self.d_model, device=self.device, dtype=self.dtype)
        
        self.lm_head = linear.Linear(
            in_features=d_model,
            out_features=vocab_size,
            device= self.device,
            dtype=self.dtype,
        )

    def forward(self, x):
        
        x = self.token_embedding(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x_norm = self.final_norm(x)
        
        logist = self.lm_head(x_norm)
        
        return logist
    
    