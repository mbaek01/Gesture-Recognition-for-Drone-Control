import os
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn.functional as F

from trainer.utils import log_metrics_to_mlflow, plot_confusion_matrix, plot_confusion_matrix_percentage, visualize_attention_heatmap


def train(model, 
          train_loader, 
          validation_loader, 
          optimizer, 
          criterion, 
          epochs, 
          num_classes,
          device,
          logger):
    
    # Early stopper
    early_stopper = EarlyStopper(logger, patience=5, min_delta=0)

    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch + 1}/{epochs} - Starting Training Phase")
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, data in enumerate(train_loader):

            data, labels = data['data'], data['label']
            labels = labels.to(device)

        
            # Convert one-hot encoded labels to class indices
            labels_onehot = F.one_hot(labels, num_classes).float()

            optimizer.zero_grad()
            output, _ = model(data, device)
            loss = criterion(output, labels_onehot)
            
            loss.backward()
            optimizer.step()

            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

 
        train_loss /= len(train_loader)

        f1 = f1_score(all_labels, all_preds, average='macro')

        # precision, recall, f1, f2 = calculate_metrics(all_labels, all_preds)
        logger.info(
            f"Epoch: {epoch + 1} - Training Completed - Loss: {train_loss:.4f}, "
            f"F1 Score: {f1: .4f}"
        )
        log_metrics_to_mlflow(epoch, 
                              "train", 
                              train_loss,
                              f1)

        # Validation Phase
        logger.info(f"Epoch: {epoch + 1} - Starting Validation Phase")
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader):
                data, labels = data['data'], data['label']
                labels = labels.to(device)

                # Convert one-hot encoded labels to class indices
                labels_onehot = F.one_hot(
                    labels, num_classes).float()

                output, _ = model(data, device)
                loss = criterion(output, labels_onehot)
                val_loss += loss.item()

                preds = torch.argmax(output, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(validation_loader)

        f1 = f1_score(all_labels, all_preds, average='macro')

        # precision, recall, f1, f2 = calculate_metrics(all_labels, all_preds)
        logger.info(
            f"Epoch: {epoch + 1} - Validation Completed - Loss: {val_loss:.4f}, "
            f"F1 Score: {f1: .4f}"
        )
        log_metrics_to_mlflow(epoch, 
                              "validation",
                              val_loss, 
                              f1)

        if early_stopper.early_stop(val_loss):
            logger.info("Early stopping triggered")
            break


def test(model, 
         test_loader, 
         criterion, 
         device, 
         num_classes,
         skip_null_class, 
         logger,
         setting,
         name):
    
    logger.info("Starting Test Phase")
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    all_attn_weights = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data, labels = data['data'], data['label']
            labels = labels.to(device)

            output, attn_weight = model(data, device)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # attn weight analysis
            all_attn_weights.append(attn_weight.detach().cpu())

    test_loss /= len(test_loader)
    f1 = f1_score(all_labels, all_preds, average='macro')

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, skip_null_class, setting, name)
    plot_confusion_matrix_percentage(cm, skip_null_class, setting, name)

    # attn weight
    cat_attn_weights = torch.cat(cat_attn_weights, dim=0)
    visualize_attention_heatmap(cat_attn_weights, 
                                ['l_cap', 'r_cap', 'l_acc', 'r_acc', 'l_gyro', 'r_gyro', 'l_quat', 'r_quat'], 
                                os.path.join(f"saved/{setting}")
                                )


    logger.info(
        f"Test Phase Completed - Average Test Loss: {test_loss:.4f}, "
        f"F1 Score: {f1: .4f}"
    )

    # Log metrics to MLflow
    log_metrics_to_mlflow(0, "test", test_loss, f1)

    return f1


class EarlyStopper:
    def __init__(self, logger, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.logger = logger

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            self.logger.info(
                f'Validation loss increased. EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
        return False