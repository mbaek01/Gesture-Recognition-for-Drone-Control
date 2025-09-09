import os
import logging
import datetime
import mlflow
import numpy as np

from config import get_args
from dataloader.utils import generate_all_sessions, generate_loso_lopo_sets
from trainer.mlflow_exp import run_mlflow_experiment
from utils import set_device, set_seed

logger = logging.getLogger(__name__)

def main(args):
    # Unused sessions and participants filtered out
    # Then grouped for LOSO and LOPO
    all_sessions = generate_all_sessions()
    loso_sets, lopo_sets = generate_loso_lopo_sets(all_sessions)

    # Seed
    set_seed(args.seed)

    # Timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{args.model}_{timestamp}"
    mlflow.set_experiment(experiment_name)

    # Set device
    device = set_device(args.gpu)

    # Log setting for the test performance
    setting_str = f"{args.model}_{args.temporal_module}_temp_agg-{args.temp_agg}_{args.fusion_method}_{args.hidden_dim}_conv-{args.num_conv_layers}_rnn-{args.num_temp_layers}_{timestamp}"
    score_log_file_path = os.path.join(args.save_path, setting_str)          
    if not os.path.exists(score_log_file_path):
            os.makedirs(score_log_file_path)

    score_log = open(os.path.join(score_log_file_path, "score.txt"), "a")

    # Train Time 
    train_time_list = []

    # LOSO
    if args.loso:
        f_macro_list_loso = []
        for i, loso_set in enumerate(loso_sets):
            train_set = loso_set['train']
            test_set = loso_set['test']
            logger.info(f"Running LOSO experiment {i + 1}/{len(loso_sets)}")
            score, time = run_mlflow_experiment(args, logger, f"LOSO_Experiment_{i + 1}", train_set, test_set, device, score_log, setting_str)
            f_macro_list_loso.append(score)
            train_time_list.append(time)

        score_log.write(f"      [LOSO] F1 Macro: mean={np.mean(f_macro_list_loso):.7f}, std={np.std(f_macro_list_loso):.7f}\n")
        score_log.write("----------------------------------------------------------------------------------------\n")
        score_log.flush()

    # LOPO
    if args.lopo:
        f_macro_list_lopo = []
        for i, lopo_set in enumerate(lopo_sets):
            train_set = lopo_set['train']
            test_set = lopo_set['test']
            logger.info(f"Running LOPO experiment {i + 1}/{len(lopo_sets)}")
            score, time = run_mlflow_experiment(args, logger, f"LOPO_Experiment_{i + 1}", train_set, test_set, device, score_log, setting_str)
            f_macro_list_lopo.append(score)
            train_time_list.append(time)

        score_log.write(f"      [LOPO] F1 Macro: mean={np.mean(f_macro_list_lopo):.7f}, std={np.std(f_macro_list_lopo):.7f}\n")
        score_log.flush()

    # Average Train Time
    avg_train_time = sum(train_time_list) / len(train_time_list)
    with open(os.path.join("saved", setting_str, "model_info.txt"), "a") as profile_log:
        profile_log.write(f"    Avg Trainig Time: {avg_train_time:.4f}" + "\n")

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
