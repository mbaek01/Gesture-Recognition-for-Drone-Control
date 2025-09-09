import os
import torch
import random
import numpy as np

def set_device(gpu_num):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")

    print(f"Device being used: {device}")

    return device


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)