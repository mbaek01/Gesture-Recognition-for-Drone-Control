import mlflow
import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataloader.utils import NormalizeSensorData
from dataloader.dataloader import SensorDataset
from model.model import get_model, get_model_profile
from trainer.trainer import train, test

NULL_CLASS = 'null_class'

def run_mlflow_experiment(args, logger, name, train_set, test_set, device, score_log, setting):

    with mlflow.start_run(run_name=name):
        mlflow.log_param("Train Set", train_set)
        mlflow.log_param("Test Set", test_set)

        # Log all parameters
        log_parameters(args, logger)
        mlflow.log_param("Skipped bad sessions", True) ## removing bad sessions

        # Dataset paths
        dataset_path = args.dataset_path

        train_sessions_to_include = train_set
        test_sessions_to_include = test_set

        transform = None

        test_dataset = SensorDataset(dataset_path = dataset_path,
                                     sessions_to_include = test_sessions_to_include,
                                     sliding_window_size = args.sliding_window_size,
                                     sliding_window_step = args.sliding_window_step,
                                     skip_null_class = args.skip_null_class,
                                     transform = transform
                                     )
        
        train_dataset = SensorDataset(dataset_path  = dataset_path,
                                      sliding_window_size = args.sliding_window_size,
                                      sessions_to_include = train_sessions_to_include,
                                      sliding_window_step = args.sliding_window_step,
                                      skip_null_class = args.skip_null_class,
                                      transform = transform
                                      )
        
        if args.normalize:
            stats = train_dataset.compute_mean_std()
            transform = NormalizeSensorData(stats)

            train_dataset.transform = transform
            test_dataset.transform = transform

            print("Normalized by train dataset")

        print("Train Dataset length:", len(train_dataset))
        print("Test Dataset Length: ", len(test_dataset) )

        ###################

        # Define the split ratio and sizes
        train_size = args.train_valid_split_ratio
        val_size = 1 - train_size
        train_dataset, val_dataset = train_test_split(
            train_dataset, test_size=val_size, random_state=42, shuffle=True)

        # Split indices

        # Create DataLoader instances
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, collate_fn=SensorDataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=True, collate_fn=SensorDataset.collate_fn)

        ###################

        # Data loaders
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=SensorDataset.collate_fn)

        # Model
        all_modalities = {'l_cap':4, 
                        'r_cap':4,
                        'l_acc': 3, 
                        'r_acc':3,
                        'l_gyro':3,
                        'r_gyro':3,
                        'l_quat':4,
                        'r_quat':4}
        filtered_modalities = [(m, all_modalities[m]) for m in args.modalities]
           
        model = get_model(args, filtered_modalities)
        model.to(device)

        # Model Info - MFLOPs, size
        get_model_profile(model, filtered_modalities, args.batch_size, device, logger, name, os.path.join("saved", setting))


        # Optimizer and loss function
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99)
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        label_map = {
            'brake': 0, 'brake_fire_left': 1, 'brake_fire_right': 2, 'come_close': 3, 'cut_engine_left': 4, 'cut_engine_right': 5,
            'down': 6, 'engine_start_left': 7, 'engine_start_right': 8, 'follow': 9, 'left': 10, 'move_away': 11, 'negative': 12,
            'release_brake': 13, 'right': 14, 'slow_down': 15, 'stop': 16, 'straight': 17, 'take_photo': 18, 'up': 19, NULL_CLASS: 20, 'claps': 21,
        }

        # Training and testing
        train_time = train(model, 
                            train_loader, 
                            val_loader,
                            optimizer, 
                            criterion, 
                            args.epochs,
                            args.num_classes,
                            device,
                            logger,
                            setting)
        
        f1, precision, recall = test(model, 
                                    args.model,
                                    test_loader, 
                                    label_map, 
                                    args.num_classes,
                                    device,
                                    args.skip_null_class,
                                    logger,
                                    setting,
                                    name)
        
        # Score log 
        metrics_str = (
            f"Test: {name} \n"
            f"    F1 Macro: {f1:.4f}\n"
            f"    Precision: {precision:.4f}\n"
            f"    Recall: {recall:.4f}\n")
        
        score_log.write(metrics_str)
        score_log.write("----------------------------------------------------------------------------------------\n")
        score_log.flush()
    return f1, precision, recall, train_time

def log_parameters(args, logger):
    """Log parameters to logger and MLflow."""
    logger.info("Logging parameters...")
    for param, value in vars(args).items():
        logger.info(f"{param}: {value}")
        mlflow.log_param(param, value)
