from collections.abc import Iterable, Callable
from typing import Optional

import math
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = {"lr": lr,
                        "betas": betas,
                        "eps": eps,
                        "weight_decay": weight_decay,
        }
        
        super().__init__(params, defaults)
        
        
    def step(self):
        for group in self.param_groups:
            
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(grad)
                    state["v"] = torch.zeros_like(grad)
                    
                m_old = state["m"]
                v_old = state["v"]
                t = state["t"]
                
                alpha_t = lr*(math.sqrt(1 - beta2 ** t))/(1 - beta1 ** t)
                m_new = beta1 * m_old + (1-beta1) * grad
                v_new = beta2 * v_old + (1-beta2) * grad**2
                state["m"] = m_new
                state["v"] = v_new
                state["t"] += 1
                p.data = p.data - alpha_t * (m_new/(torch.sqrt(v_new) + eps))
                p.data = p.data - lr * weight_decay * p.data