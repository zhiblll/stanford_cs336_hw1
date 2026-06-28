import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        
        self.weight: nn.Parameter = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device = device, dtype = dtype))
        self.reset_parameters()
    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, std = 1.0, a = -3, b = 3)
    
    def forward(self, x) -> torch.tensor:
        return self.weight[x]
    
        
        
    
         