import mlflow

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataloader.utils import NormalizeSensorData
from dataloader.dataloader import SensorDataset
from model.model import get_model
from trainer.trainer import train, test


def run_mlflow_experiment(args, logger, name, train_set, test_set, device):

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

    model = get_model(args)
    model.to(device)

    ###################


    # Optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Training and testing
    train(model, 
          train_loader, 
          val_loader,
          optimizer, 
          criterion, 
          args.epochs,
          args.num_classes,
          device,
          logger)
    
    test(model, 
         test_loader, 
         criterion, 
         device,
         args.num_classes,
         args.skip_null_class,
         logger)

def log_parameters(args, logger):
    """Log parameters to logger and MLflow."""
    logger.info("Logging parameters...")
    for param, value in vars(args).items():
        logger.info(f"{param}: {value}")
        mlflow.log_param(param, value)
