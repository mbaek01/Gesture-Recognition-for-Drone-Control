import os
import random
import torch
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.colors as mcolors

label_map = {
    'brake': 0, 'brake_fire_left': 1, 'brake_fire_right': 2, 'come_close': 3, 'cut_engine_left': 4, 'cut_engine_right': 5,
    'down': 6, 'engine_start_left': 7, 'engine_start_right': 8, 'follow': 9, 'left': 10, 'move_away': 11, 'negative': 12,
    'release_brake': 13, 'right': 14, 'slow_down': 15, 'stop': 16, 'straight': 17, 'take_photo': 18, 'up': 19, 'NULL_CLASS': 20, 'claps': 21,
}


def log_metrics_to_mlflow(epoch, phase, loss, f1):
    """
    Log metrics to MLflow.
    :param epoch: Current epoch
    :param phase: 'train', 'validation', or 'test'
    :param loss: Loss value
    :param precision: Precision score
    :param recall: Recall score
    :param f1: F1 score
    :param f2: F2 score
    """
    mlflow.log_metric(f"{phase}_loss", loss, step=epoch)
    # mlflow.log_metric(f"{phase}_precision", precision, step=epoch)
    # mlflow.log_metric(f"{phase}_recall", recall, step=epoch)
    mlflow.log_metric(f"{phase}_f1_score", f1, step=epoch)
    # mlflow.log_metric(f"{phase}_f2_score", f2, step=epoch)


def plot_confusion_matrix(cm, skip_null_class, path, name):
    # Filter labels
    filtered_items = [(label, idx) for label, idx in label_map.items()
                      if label != 'claps' and (label != 'NULL_CLASS' or not skip_null_class)]

    # Sort by index
    filtered_items = sorted(filtered_items, key=lambda x: x[1])
    label_names = [label for label, _ in filtered_items]
    label_indices = [idx for _, idx in filtered_items]

    # Filter the confusion matrix to only include selected indices
    cm_filtered = cm[np.ix_(label_indices, label_indices)]

    # Plot
    plt.figure(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_filtered, display_labels=label_names)
    disp.plot(include_values=True, cmap='Blues',
              xticks_rotation=90, ax=plt.gca())
    
    n_classes = len(label_names)

    # Make diagonal values bold and white
    for i, text in enumerate(disp.text_.ravel()):
        row = i // n_classes
        col = i % n_classes
        if row == col:  # diagonal only
            text.set_color("white")
            text.set_fontweight("bold")

    plt.title('Confusion Matrix')
    plt.tight_layout()

    plt_path = os.path.join(path, f"conf_matrix.png")
    plt.savefig(plt_path)
    mlflow.log_artifact(plt_path)
    plt.close()

def plot_confusion_matrix_percentage(cm, skip_null_class, path, name):
        # Filter labels
     # Filter labels
    filtered_items = [(label, idx) for label, idx in label_map.items()
                      if label != 'claps' and (label != 'NULL_CLASS' or not skip_null_class)]

    # Sort by index
    filtered_items = sorted(filtered_items, key=lambda x: x[1])
    label_names = [label for label, _ in filtered_items]
    label_indices = [idx for _, idx in filtered_items]

    # Filter the confusion matrix to only include selected indices
    cm_filtered = cm[np.ix_(label_indices, label_indices)]

    # Calculate percentages
    cm_sum = cm_filtered.sum(axis=1)[:, np.newaxis]
    cm_percentage = np.divide(cm_filtered, cm_sum, where=cm_sum != 0) * 100

    # Plot using percentages
    plt.figure(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=label_names)
    disp.plot(include_values=False, cmap='Blues', xticks_rotation=90, ax=plt.gca(), colorbar=True)
    
    # Manually add text
    ax = plt.gca()

    for (i, j), val in np.ndenumerate(cm_percentage):
        if not np.isnan(val):
            if i == j:  # diagonal cell
                ax.text(j, i, f"{val:.1f}%",
                        ha='center', va='center',
                        color='white', fontsize=9, fontweight='bold')
            else:  # off-diagonal
                ax.text(j, i, f"{val:.1f}%",
                        ha='center', va='center',
                        color='black', fontsize=8)

    plt.title('Confusion Matrix (Percentages)')
    plt.tight_layout()

    plt_path = os.path.join(path, f"conf_matrix_pct.png")
    plt.savefig(plt_path)  
    mlflow.log_artifact(plt_path)  
    plt.close()


def visualize_attention_heatmap(attention_matrix, modalities, label_str, file_path=None):
    """
    Visualizes an attention matrix as a heatmap.

    Args:
        attention_matrix (np.ndarray): The 2D array of attention weights.
                                     Shape: (num_modalities, num_modalities)
        modalities (list): A list of strings for the modality labels.
        file_path (str, optional): The path to save the plot. 
                                   If None, the plot is not saved.
    """
    if attention_matrix.ndim != 2:
        raise ValueError("Input attention_matrix must be a 2D array.")

    num_modalities = len(modalities)
    if attention_matrix.shape != (num_modalities, num_modalities):
        raise ValueError("Shape of attention_matrix must match the number of modalities.")

    # Create the heatmap plot
    plt.figure(figsize=(10, 8))

    # cmap = plt.cm.get_cmap("inferno")
    # shifted_cmap = mcolors.LinearSegmentedColormap.from_list(
    #     "shifted_inferno", cmap(np.linspace(0.3, 1, 256))
    # )

    # plt.imshow(attention_matrix, cmap=shifted_cmap)
    plt.imshow(attention_matrix, cmap="viridis")
    
    # Set the title and labels
    plt.title(f'Attention Weights Heatmap - {label_str}')
    plt.xlabel('Keys (Modalities Paying Attention)')
    plt.ylabel('Queries (Modalities Receiving Attention)')
    
    # Set the ticks and labels for the axes
    plt.xticks(np.arange(num_modalities), modalities, rotation=45, ha='right')
    plt.yticks(np.arange(num_modalities), modalities)
    
    # Add a color bar and adjust layout
    plt.colorbar(label='Attention Weight')
    plt.tight_layout()
    
    # Show the plot
    plt.show()

    # Save the plot to a file if a file path is provided
    if file_path:
        plt.savefig(file_path)

def attention_heatmap_per_label(all_attn_weights, all_labels, all_preds, label_map, modalities, vis_path):
    # attn weight
    cat_attn_weights = torch.cat(all_attn_weights, dim=0)

    # attn weight idx per label (when correctly labeled)
    idx_to_label = {v:k for k,v in label_map.items()}
    correct_label_idx = {k:[] for k in label_map.keys()}

    for i, (label, pred) in enumerate(zip(all_labels, all_preds)):
        if label == pred:
            label_str = idx_to_label[pred]
            correct_label_idx[label_str].append(i)
    
    # Random sampling of attention weight indices
    sampled_label_idx = {k: random.sample(v, 20) for k,v in correct_label_idx}

    modalities = [mod_name for mod_name, _ in modalities]

    for label_str in sampled_label_idx.keys():
        idx_list = sampled_label_idx[label_str]
        for i in idx_list:
            curr_path = os.path.join(vis_path, label_str)
            if not os.path.exists(curr_path):
                os.makedir(curr_path)
                
            visualize_attention_heatmap(cat_attn_weights[i],
                                        label_str,
                                        modalities,
                                        os.path.join(curr_path, f"attn_weight_{label_str}_{i}.png"))
            

def plot_avg_contributions(llrs_dict, label_map, num_classes, save_path, logger):
    """
    Plots and saves average modality contributions to class predictions using
    the inverted magnitude method and different colors for each modality.
    
    Args:
        llrs_dict (dict): Dictionary of modality -> (all_llrs, all_labels)
        label_map (dict): Dictionary mapping string labels to integer indices.
        num_classes (int): The total number of classes.
        save_path (str): Path to the directory where plots will be saved.
        logger: A logger object for logging information.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    idx_to_label = {v:k for k,v in label_map.items()}

    # Calculate mean LLR per modality, per class
    avg_contrib = {modality: np.zeros(num_classes) for modality in llrs_dict}

    for modality, (all_llrs, all_labels) in llrs_dict.items():
        for c in range(num_classes):
            mask = all_labels == c
            if mask.sum() > 0:
                avg_contrib[modality][c] = all_llrs[mask, c].mean()

    modalities = list(avg_contrib.keys())

    # cmap per modality
    cmap = plt.cm.get_cmap('viridis', len(modalities))
    colors = [cmap(i) for i in range(len(modalities))]

    for c in range(num_classes):
        class_name = idx_to_label.get(c, f"Unknown Class {c}")
        
        contributions = [avg_contrib[mod][c] for mod in modalities]
        
        if np.sum(np.abs(contributions)) == 0:
            logger.info(f"Skipping plot for class '{class_name}' as there is no data.")
            continue

        magnitudes = np.abs(np.array(contributions))
        epsilon = 1e-9
        contribution_scores = 1 / (magnitudes + epsilon)

        plt.figure(figsize=(8, 5))
        
        plt.bar(modalities, contribution_scores, color=colors)
        
        plt.xlabel("Modality")
        plt.ylabel("Contribution Score (1 / LLR Magnitude)")
        plt.title(f"Modality Contribution Score for Class: '{class_name}'")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        safe_class_name = class_name.replace(" ", "_").replace("/", "_")
        plot_filename = os.path.join(save_path, f"avg_contrib_{safe_class_name}.png")
        plt.savefig(plot_filename, bbox_inches="tight")
        plt.close()