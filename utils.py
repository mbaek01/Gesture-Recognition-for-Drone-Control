import os
import json
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


def save_config_json(args, save_path):
    model_args = {
        "num_conv_layers": args.num_conv_layers,
        "num_temp_layers": args.num_temp_layers,
        "temporal_module": args.temporal_module,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "kernel_size": args.kernel_size,
        "modalities": args.modalities
                }

    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "model_args.json")

    # Save the dictionary to a file as a nicely formatted JSON
    with open(file_path, "w") as f:
        json.dump(model_args, f, indent=4)


def log_metric_result(log_path, metric_list, exp_mode):
    f1_list = [score[0] for score in metric_list]
    precision_list = [score[1] for score in metric_list]
    recall_list = [score[2] for score in metric_list]
    time_list = [score[3] for score in metric_list]

    with open(log_path, "a") as score_log:
        score_log.write(f"      [{exp_mode}] F1 Macro:  mean={np.mean(f1_list):.4f}, std={np.std(f1_list):.4f}\n"
                        f"             Precision: mean={np.mean(precision_list):.4f}, std={np.std(precision_list):.4f}\n"
                        f"             Recall:    mean={np.mean(recall_list):.4f}, std={np.std(recall_list):.4f}\n"
                        f"             Time:      mean={np.mean(time_list):.4f}, std={np.std(time_list):.4f}\n")
        score_log.write("----------------------------------------------------------------------------------------\n")
        score_log.flush()
