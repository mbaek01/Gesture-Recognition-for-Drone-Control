import os
import logging
import datetime
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
    high_level_folder = f"./mlruns/{args.model}_{timestamp}"
    os.makedirs(high_level_folder, exist_ok=True)
    
    # Set device
    device = set_device()

    if args.loso:
        for i, loso_set in enumerate(loso_sets):
            train_set = loso_set['train']
            test_set = loso_set['test']
            logger.info(f"Running LOSO experiment {i + 1}/{len(loso_sets)}")

            # mlflow path
            experiment_name = f"LOSO_Experiment_{i + 1}"
            artifact_path = os.path.join(high_level_folder, experiment_name)

            run_mlflow_experiment(args, logger, experiment_name, artifact_path, train_set, test_set, device)

    if args.lopo:
        for i, lopo_set in enumerate(lopo_sets):
            train_set = lopo_set['train']
            test_set = lopo_set['test']
            logger.info(f"Running LOPO experiment {i + 1}/{len(lopo_sets)}")
            run_mlflow_experiment(args, logger, f"LOPO_Experiment_{i + 1}", train_set, test_set, device)


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
