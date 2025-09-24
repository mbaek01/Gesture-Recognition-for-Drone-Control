import os
import logging
import datetime
import mlflow
import numpy as np
import itertools

from config import get_args
from dataloader.utils import generate_all_sessions, generate_loso_lopo_sets
from trainer.mlflow_exp import run_mlflow_experiment
from utils import set_device, set_seed, save_config_json, log_metric_result

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
    # setting_str = f"{args.model}_{args.temporal_module}_temp_agg-{args.temp_agg}_{args.fusion_method}_{args.hidden_dim}_conv-{args.num_conv_layers}_rnn-{args.num_temp_layers}_{timestamp}"
    
    modalities_str = "_".join(args.modalities)
    setting_str = f"{args.model}_{modalities_str}_{timestamp}"
    
    score_log_file_path = os.path.join(args.save_path, setting_str)          
    if not os.path.exists(score_log_file_path):
        os.makedirs(score_log_file_path)

    score_log = open(os.path.join(score_log_file_path, "score.txt"), "a")

    # Save config
    save_config_json(args, score_log_file_path)

    # # LOSO
    # if args.loso:
    #     score_list_loso = []
    #     for i, loso_set in enumerate(loso_sets):
    #         train_set = loso_set['train']
    #         test_set = loso_set['test']
    #         logger.info(f"Running LOSO experiment {i + 1}/{len(loso_sets)}")
    #         f1, precision, recall, time = run_mlflow_experiment(args, logger, f"LOSO_Experiment_{i + 1}", train_set, test_set, device, score_log, setting_str)
    #         score_list_loso.append((f1, precision, recall, time))
        
    #     log_metric_result(os.path.join("saved", setting_str, "score.txt"), score_list_loso, "LOSO")

    score_list_loso = []
    # LOPO
    if args.lopo:
        score_list_lopo = []
        for i, lopo_set in enumerate(lopo_sets):
            if i!=5: continue
            train_set = lopo_set['train']
            test_set = lopo_set['test']
            logger.info(f"Running LOPO experiment {i + 1}/{len(lopo_sets)}")
            f1, precision, recall, time = run_mlflow_experiment(args, logger, f"LOPO_Experiment_{i + 1}", train_set, test_set, device, score_log, setting_str)
            score_list_lopo.append((f1, precision, recall, time))

        log_metric_result(os.path.join("saved", setting_str, "score.txt"), score_list_lopo, "LOPO")


    # Average Train Time
    all_train_time = [score[3] for score in itertools.chain(score_list_lopo, score_list_loso)]
    with open(os.path.join("saved", setting_str, "model_info.txt"), "a") as profile_log:
        profile_log.write(f"    Train Time: mean={np.mean(all_train_time):.4f} min, std={np.std(all_train_time):.4f}\n")
        profile_log.flush()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
