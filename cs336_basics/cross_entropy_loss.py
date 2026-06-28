import torch

def cross_entropy(logist, target):
    
    max_value = logist.max(dim = -1, keepdim = True).values
    
    shifted_logist = logist - max_value
    exp_logist = torch.exp(shifted_logist)
    
    sum_exp = exp_logist.sum(dim = -1)
    
    log_sum_exp = torch.log(sum_exp)
    # Wrong:
    # logist = logist - max_value
    # log_logist_exp = torch.log(exp_logist)
    # prediction = torch.gather(log_logist_exp, dim = -1, index = target.unsqueeze(-1))
    #
    # torch.log(torch.exp(shifted_logist)) is numerically unstable: very negative
    # entries underflow to 0, then log(0) becomes -inf.
    prediction = torch.gather(shifted_logist, dim = -1, index = target.unsqueeze(-1))
    loss = log_sum_exp.unsqueeze(-1) - prediction

    return loss.mean()
    
    
