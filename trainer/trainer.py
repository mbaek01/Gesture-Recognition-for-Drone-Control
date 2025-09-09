import os
import numpy as np
import random
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn.functional as F
from collections import Counter

from trainer.utils import log_metrics_to_mlflow, plot_confusion_matrix, plot_confusion_matrix_percentage, attention_heatmap_per_label, plot_avg_contributions


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
         model_name,
         test_loader, 
         label_map, 
         num_classes,
         device, 
         skip_null_class, 
         logger,
         setting,
         name):
    
    logger.info("Starting Test Phase")
    model.eval()
    all_preds = []
    all_labels = []

    # saved for ablation study
    if model_name == "feature_fusion":
        all_attn_weights = []

    elif model_name == "llr_fusion":
        correct_llrs = {modality_name: [] for modality_name, _ in model.modalities}
        

    # Test
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data, labels = data['data'], data['label']
            labels = labels.to(device)

            output, sub = model(data, device)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            # attn weight analysis
            if model_name == "feature_fusion":
                all_attn_weights.append(sub.detach().cpu())

            # llr analysis
            elif model_name == "llr_fusion":
                correct_mask = (preds == labels)

                for modality_name, llr_tensor in sub.items():
                    if correct_mask.any():
                        # filter both logits and labels with the same mask
                        filtered_llrs   = llr_tensor[correct_mask].detach().cpu().numpy()
                        filtered_labels = labels[correct_mask]

                        # save as (llrs, labels) tuple
                        correct_llrs[modality_name].append((filtered_llrs, filtered_labels))

            all_preds.extend(preds)
            all_labels.extend(labels)

    f1 = f1_score(all_labels, all_preds, average='macro')

    
    # if model_name == "llr_fusion": # for random sampling
    #     for modality_name in correct_llrs.keys():
    #         random.shuffle(correct_llrs[modality_name])
    

    # confusion matrix
    vis_path = os.path.join(f"saved/{setting}", name)

    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    cm = confusion_matrix(all_labels, all_preds)

    plot_confusion_matrix(cm, skip_null_class, vis_path, name)
    plot_confusion_matrix_percentage(cm, skip_null_class, vis_path, name)

    # LLR Fusion: LLR contribution in class pred
    if model_name == "llr_fusion":

        # concatenate all tuples into tensors per modality
        for modality_name in correct_llrs:
            if correct_llrs[modality_name]:
                llrs_list, labels_list = zip(*correct_llrs[modality_name])
                correct_llrs[modality_name] = (
                    np.concatenate(llrs_list, axis=0),
                    np.concatenate(labels_list, axis=0)
                )

        # plot contributions
        plot_avg_contributions(correct_llrs, label_map, num_classes, vis_path, logger)

    # Feature Fusion: attention weight heat map
    if model_name == "feature_fusion":
        attention_heatmap_per_label(all_attn_weights, all_labels, all_preds, label_map, model.modalities, vis_path)


    logger.info(
        f"Test Phase Completed - F1 Score: {f1: .4f}"
    )

    # Log metrics to MLflow
    log_metrics_to_mlflow(0, "test", 0, f1)

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