import torch
from torch import nn


class RMSNorm(nn.Module):
    # 原代码：
    # def __init__(self, d_model: int, eps, device: torch.device, dtype: torch.dtype):
    # 改成下面这样：
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # 原代码：
        # self.gain = nn.Parameter(torch.empty((self.d_model,), device = device, d_type = dtype))
        # 改动原因：
        # 1. d_type 拼错了，应该是 dtype
        # 2. 这里 shape=(d_model,) 是对的，保留
        self.weight = nn.Parameter(
            torch.empty((self.d_model,), device=device, dtype=dtype)
        )

        # 新增：
        # 原代码里虽然写了 reset_param，但 __init__ 里没有调用
        # 这样 gain 会保持未初始化状态，所以这里要主动调用
        self.reset_parameters()

    # 原代码：
    # def reset_param(self):
    #     self.gain = torch.ones_like(self.gain)
    #
    # 改成下面这样：
    def reset_parameters(self) -> None:
        # 改动原因：
        # 1. 方法名改成 reset_parameters，更符合 PyTorch 习惯
        # 2. 不能直接 self.gain = ...，否则会把 nn.Parameter 替换成普通 Tensor
        # 3. 应该原地把参数填成 1
        with torch.no_grad():
            self.weight.fill_(1.0)

    # 原代码：
    # def forward(self,x):
    #     a = torch.pow(x,2)
    #     a = a.sum(dim = -1)
    #     b = b + self.eps
    #     b = 1/b
    #     return self.gain*x*b
    #
    # 改成下面这样：
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 改动原因：
        # 作业要求先 upcast 到 float32，再做 RMSNorm，最后 cast 回原 dtype
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # 原代码：
        # a = torch.pow(x, 2)
        # a = a.sum(dim=-1)
        #
        # 改动原因：
        # 1. RMSNorm 要的是 mean，不是 sum
        # 2. keepdim=True，这样结果 shape 是 (..., 1)，后面才能和 x 广播相乘
        mean_square = torch.pow(x, 2).mean(dim=-1, keepdim=True)

        # 原代码：
        # b = b + self.eps
        # b = 1 / b
        #
        # 改动原因：
        # 1. 你这里 b 变量根本没定义
        # 2. RMS 需要 sqrt，所以更自然的写法是 rsqrt(mean_square + eps)
        inv_rms = torch.rsqrt(mean_square + self.eps)

        # 原代码：
        # return self.gain * x * b
        #
        # 改动原因：
        # 1. 这里应该乘 inv_rms，不是 b
        # 2. self.gain shape=(d_model,) 会自动广播到 x 的最后一维
        # 3. 最后 cast 回输入原始 dtype
        result = x * inv_rms * self.weight.to(torch.float32)
        return result.to(in_dtype)