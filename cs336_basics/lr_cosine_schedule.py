import math
def get_lr_cosine_schedule(
    t: int,
    alpha_max,
    alpha_min,
    t_w,
    t_c):
    
    if t < t_w:
        lr = (t/t_w)*alpha_max
        
    elif t_w <= t <= t_c: 
        lr =  alpha_min + 0.5*(1 + math.cos((t - t_w)*math.pi/(t_c - t_w)))*(alpha_max - alpha_min)
        
    else: 
        lr = alpha_min
    
    return lr

        