import torch
import numpy as np




def get_batch(x, length, batch_num, device):
    
    input_seq = []
    target_seq = []
    
    max_start_pos = len(x) - 1 - length
    
    starts = np.random.randint(0, max_start_pos + 1, size=batch_num)
    
    for start in starts:
        end = start + length
        input_seq.append(x[start:end])
        target_seq.append(x[start +1:end +1])
    
    inputs_np = np.stack(input_seq, axis=0)
    targets_np = np.stack(target_seq, axis=0)
    
    inputs = torch.tensor(inputs_np, dtype=torch.long, device=device)
    targets = torch.tensor(targets_np, dtype=torch.long, device=device)

    return inputs, targets