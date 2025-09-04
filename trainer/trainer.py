import numpy as np

from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn.functional as F

from trainer.utils import log_metrics_to_mlflow, plot_confusion_matrix, plot_confusion_matrix_percentage


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
            output = model(data, device)
            loss = criterion(output, labels_onehot)
            
            loss.backward()
            optimizer.step()

            # train_loss += loss.item()
            # running_loss = train_loss / (batch_idx + 1)

            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # print(labels_indices.shape, output.shape)
            # print(labels_indices)
            # print(output)

            # # Batch-level metrics
            # batch_precision, batch_recall, batch_f1, _ = calculate_metrics(
            #     labels.cpu().numpy(), preds)

            # # Log batch-level details
            # logger.info(
            #     f"Epoch {epoch + 1}, Batch {batch_idx +1}/{len(train_loader)} - "f"Loss: {loss.item(): .4f}, Running Avg Loss: {running_loss: .4f}, "
            #     f"Precision: {batch_precision: .4f}, Recall: {batch_recall: .4f}, F1: {batch_f1: .4f}"
            # )

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

                output = model(data, device)
                loss = criterion(output, labels_onehot)
                val_loss += loss.item()
                # running_val_loss = val_loss / (batch_idx + 1)

                preds = torch.argmax(output, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

                # # Batch-level metrics
                # batch_precision, batch_recall, batch_f1, _ = calculate_metrics(
                #     labels.cpu().numpy(), preds)

                # # Log batch-level details
                # logger.info(
                #     f"Epoch {epoch + 1}, Batch {batch_idx +1}/{len(validation_loader)} - "f"Validation Loss: {loss.item(): .4f}, Running Avg Loss: {running_val_loss: .4f}, "f"Precision: {batch_precision: .4f}, Recall: {batch_recall: .4f}, F1: {batch_f1: .4f}"
                # )

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
         logger):
    
    logger.info("Starting Test Phase")
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data, labels = data['data'], data['label']
            labels = labels.to(device)

            # Convert one-hot encoded labels to class indices
            labels_onehot = F.one_hot(
                labels, num_classes).float()

            output = model(data, device)
            loss = criterion(output, labels_onehot)
            test_loss += loss.item()
            running_test_loss = test_loss / (batch_idx + 1)

            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # # Batch-level metrics
            # batch_precision, batch_recall, batch_f1, _ = calculate_metrics(
            #     labels.cpu().numpy(), preds)

            # # Log batch-level details
            # logger.info(
            #     f"Batch {batch_idx + 1}/{len(test_loader)} - Test Loss: {loss.item(): .4f}, "f"Running Avg Loss: {running_test_loss: .4f}, Precision: {batch_precision: .4f}, "f"Recall: {batch_recall:.4f}, F1: {batch_f1:.4f}"
            # )

    test_loss /= len(test_loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    # precision, recall, f1, f2 = calculate_metrics(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, skip_null_class)
    plot_confusion_matrix_percentage(cm, skip_null_class)
    logger.info(
        f"Test Phase Completed - Average Test Loss: {test_loss:.4f}, "
        f"F1 Score: {f1: .4f}"
    )

    # Log metrics to MLflow
    log_metrics_to_mlflow(0, "test", test_loss, f1)


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