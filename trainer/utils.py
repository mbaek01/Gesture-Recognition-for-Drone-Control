import os
import random
import torch
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.colors as mcolors
from sklearn.metrics import f1_score, precision_score, recall_score

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
    ax = plt.gca()  # Get the current axis
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_filtered, display_labels=label_names)
    
    # --- START OF MODIFIED BLOCK ---
    # 1. Plot *without* the automatic colorbar
    disp.plot(include_values=True, cmap='Blues',
              xticks_rotation=90, ax=ax, colorbar=False) # Ensure colorbar is off
    
    # 2. Manually add the colorbar, explicitly passing 'ax=ax'
    #    This forces the colorbar to match the height of the 'ax' object.
    plt.colorbar(disp.im_, ax=ax)
    # --- END OF MODIFIED BLOCK ---
    
    n_classes = len(label_names)

    # Make diagonal values bold and larger
    for i, text in enumerate(disp.text_.ravel()):
        row = i // n_classes
        col = i % n_classes
        if row == col:  # diagonal only
            text.set_fontweight("bold")
            text.set_fontsize(23)
        else:
            text.set_fontsize(18)

    # Apply font and label changes
    # plt.title('Confusion Matrix (counts)', fontsize=22, fontname='Times New Roman') # Removed title
    ax.set_xlabel("")  # Removed x-axis label
    ax.set_ylabel("")  # Removed y-axis label
    
    # Remove x-tick labels, increase y-tick label font size
    ax.set_xticklabels([])
    plt.yticks(fontsize=22)
    
    # Set y-tick label font
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    plt.tight_layout()

    # --- UPDATED SAVING BLOCK ---
    # Save as PNG
    png_path = os.path.join(path, f"conf_matrix.png")
    plt.savefig(png_path)
    mlflow.log_artifact(png_path)
    
    # Save as PDF
    pdf_path = os.path.join(path, f"conf_matrix.pdf")
    plt.savefig(pdf_path, format='pdf')
    mlflow.log_artifact(pdf_path)
    # --- END UPDATED BLOCK ---
    
    plt.close()

# No new imports are needed for this method

def plot_confusion_matrix_percentage(cm, skip_null_class, path, name):
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
    cm_percentage = np.divide(cm_filtered, cm_sum,
                              out=np.full_like(cm_filtered, np.nan, dtype=float),
                              where=cm_sum != 0) * 100

    # Plot using percentages
    plt.figure(figsize=(14, 12))
    ax = plt.gca()  # Get the current axis
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=label_names)

    # --- START OF NEW METHOD ---
    # 1. Plot *without* the automatic colorbar
    disp.plot(include_values=False, cmap='Blues', xticks_rotation=90, ax=ax, colorbar=False)
    
    # 2. Manually add the colorbar, explicitly passing 'ax=ax'
    #    This forces the colorbar to match the height of the 'ax' object.
    plt.colorbar(disp.im_, ax=ax)
    # --- END OF NEW METHOD ---
    
    # Manually add text with contrastive colors
    cmap = disp.im_.cmap
    norm = disp.im_.norm

    for (i, j), val in np.ndenumerate(cm_percentage):
        if not np.isnan(val):
            # Determine background color and luminance
            bg_color = cmap(norm(val))
            luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
            text_color = 'white' if luminance < 0.5 else 'black'

            if i == j:  # diagonal cell
                ax.text(j, i, f"{val:.0f}",
                        ha='center', va='center',
                        color=text_color, fontsize=23, fontweight='bold')
            else:  # off-diagonal
                ax.text(j, i, f"{val:.0f}",
                        ha='center', va='center',
                        color=text_color, fontsize=18)

    # Apply font and label changes
    # plt.title('Confusion Matrix (%)', fontsize=22, fontname='Times New Roman') # Removed title
    ax.set_xlabel("")  # Removed x-axis label
    ax.set_ylabel("")  # Removed y-axis label

    # Remove x-tick labels, increase y-tick label font size
    ax.set_xticklabels([])
    plt.yticks(fontsize=22)
    
    # Set y-tick label font
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    plt.tight_layout()

    # --- UPDATED SAVING BLOCK ---
    # Save as PNG
    png_path = os.path.join(path, f"conf_matrix_pct.png")
    plt.savefig(png_path)  
    mlflow.log_artifact(png_path)
    
    # Save as PDF
    pdf_path = os.path.join(path, f"conf_matrix_pct.pdf")
    plt.savefig(pdf_path, format='pdf')
    mlflow.log_artifact(pdf_path)
    # --- END UPDATED BLOCK ---
      
    plt.close()


def visualize_attention_heatmap(attention_matrix, label_str, modalities, i, file_path=None):
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
    plt.imshow(attention_matrix, cmap="viridis")

    # Set the title and labels
    # plt.title(f'Attention Weights Heatmap for {label_str} - Random Sample #{i}', fontsize=18)
    plt.xlabel('Attended Modality (Key)', fontsize=22)
    plt.ylabel('Receiving Modality (Query)', fontsize=22)

    # Set the ticks and labels for the axes
    plt.xticks(np.arange(num_modalities), modalities, rotation=45, ha='right', fontsize=22)
    plt.yticks(np.arange(num_modalities), modalities, fontsize=22)

    # Add a color bar and adjust layout
    cbar = plt.colorbar(label='Attention Weight')
    cbar.set_label('Attention Weight', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    plt.tight_layout()

    # Save the plot to a file if a file path is provided
    if file_path:
        if not file_path.lower().endswith(".pdf"):
            file_path = file_path.rsplit(".", 1)[0] + ".pdf"
        plt.savefig(file_path, bbox_inches="tight", format="pdf")

    plt.close()

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
    sampled_label_idx = {k: random.sample(v, min(10, len(v))) for k,v in correct_label_idx.items()}

    modalities = [mod_name for mod_name, _ in modalities]

    for label_str in sampled_label_idx.keys():
        idx_list = sampled_label_idx[label_str]
        for i in idx_list:
            curr_path = os.path.join(vis_path, label_str)
            if not os.path.exists(curr_path):
                os.makedirs(curr_path)
                
            visualize_attention_heatmap(cat_attn_weights[i],
                                        label_str,
                                        modalities,
                                        i,
                                        os.path.join(curr_path, f"attn_weight_{label_str}_{i}.pdf"),
                                        )


def _plot_contribution_chart(modalities, scores, colors, title, ylabel, save_path, filename):
    """
    A helper function that creates and saves the contribution bar chart.
    
    Args:
        modalities (list): List of modality names for the x-axis.
        scores (list): List of contribution scores for the y-axis.
        colors (list): List of colors for the bars.
        title (str): The title of the plot.
        ylabel (str): The label for the y-axis.
        save_path (str): The directory to save the plot in.
        filename (str): The name of the file to save (without extension).
    """
    plt.figure(figsize=(8, 5))
    plt.bar(modalities, scores, color=colors)
    
    plt.xlabel("Modality", fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    # plt.title(title, fontsize=22)

    plt.xticks(rotation=45, ha="right", fontsize=22)
    plt.yticks(fontsize=20)

    plt.grid(axis='y', linestyle='--', alpha=0.7)    
    # Sanitize the filename and create the full save path
    safe_filename = filename.replace(" ", "_").replace("/", "_")
    plot_filepath = os.path.join(save_path, f"{safe_filename}.png")
    
    # plt.savefig(plot_filepath, bbox_inches="tight")

    plot_filepath_pdf = os.path.join(save_path, f"{safe_filename}.pdf")
    plt.savefig(plot_filepath_pdf, bbox_inches="tight")

    plt.close()


def plot_avg_contributions(llrs_dict, label_map, num_classes, save_path, logger):
    """
    Plots and saves average modality contributions by calling a plotting helper function.
    """
    os.makedirs(save_path, exist_ok=True)
    idx_to_label = {v: k for k, v in label_map.items()}

    # --- Data Preparation (No Changes Here) ---
    avg_contrib = {modality: np.zeros(num_classes) for modality in llrs_dict}
    for modality, (all_llrs, all_labels) in llrs_dict.items():
        for c in range(num_classes):
            mask = all_labels == c
            if mask.sum() > 0:
                avg_contrib[modality][c] = all_llrs[mask, c].mean()

    # 1. Create a sorted, canonical list of ALL possible modalities. Sorting is key!
    canonical_modalities = ['l_cap', 'r_cap', 'l_acc', 'r_acc', 'l_gyro', 'r_gyro', 'l_quat','r_quat']
    
    # 2. Create a persistent mapping (dictionary) from the name to a color
    cmap = plt.cm.get_cmap('plasma', len(canonical_modalities))
    modality_color_map = {mod_name: cmap(i) for i, mod_name in enumerate(canonical_modalities)}

    # 3. Get the list of modalities for this calculation (the order matters for the bars)
    modalities = list(avg_contrib.keys()) # This is the order they appear on the chart
    
    # 4. Build the FINAL colors list by LOOKING UP each modality in the stable map
    colors = [modality_color_map[mod] for mod in modalities]

    # --- Plotting Loop ---
    for c in range(num_classes):
        class_name = idx_to_label.get(c, f"Unknown Class {c}")
        contributions = [avg_contrib[mod][c] for mod in modalities]
        
        if np.sum(np.abs(contributions)) == 0:
            logger.info(f"Skipping plot for class '{class_name}' as there is no data.")
            continue

        magnitudes = np.abs(np.array(contributions))
        contribution_scores = 1 / (magnitudes + 1e-9)

        # --- THIS IS THE CHANGE ---
        # The entire plt.figure... block is replaced by this single call
        _plot_contribution_chart(
            modalities=modalities,
            scores=contribution_scores,
            colors=colors,
            title=f"Avg. Modality Contribution for Class: '{class_name}'",
            ylabel="Avg. Contribution (1 / |LLR|)",
            save_path=os.path.join(save_path,class_name),
            filename=f"avg_contrib_{class_name}"
        )


def plot_n_random_samples_per_class(llrs_dict, label_map, num_classes, save_path, logger, n=10):
    """
    Finds N random samples for each class and plots their individual
    modality contributions.
    """
    os.makedirs(save_path, exist_ok=True)
    idx_to_label = {v: k for k, v in label_map.items()}
    modalities = list(llrs_dict.keys())

    # We can use the data from any modality to get the total number of samples and labels
    first_modality = modalities[0]
    num_samples = len(llrs_dict[first_modality][0])
    all_labels = llrs_dict[first_modality][1]

    # Random Sampling  
    all_indices = list(range(num_samples))
    random.shuffle(all_indices)

    # 1. Create a sorted, canonical list of ALL possible modalities.
    canonical_modalities = ['l_cap', 'r_cap', 'l_acc', 'r_acc', 'l_gyro', 'r_gyro', 'l_quat','r_quat']
    
    # 2. Create the persistent mapping
    cmap = plt.cm.get_cmap('plasma', len(canonical_modalities))
    modality_color_map = {mod_name: cmap(i) for i, mod_name in enumerate(canonical_modalities)}

    # 3. Build the colors list by looking up colors for the calculation list ('modalities')
    colors = [modality_color_map[mod] for mod in modalities]

    # Loop through each class
    for c in range(num_classes):
        class_name = idx_to_label.get(c, f"Unknown_Class_{c}")
        logger.info(f"Processing class: '{class_name}'")
        
        # Find the indices of samples belonging to this class FROM THE SHUFFLED LIST
        # This gives us a random ordering of samples for this class
        class_indices_random_order = [idx for idx in all_indices if all_labels[idx] == c]
        
        if len(class_indices_random_order) == 0:
            logger.info(f"-> No samples found for class '{class_name}'.")
            continue
            
        # Get the first N indices from our randomly ordered list
        indices_to_plot = class_indices_random_order[:n]
        logger.info(f"-> Found {len(class_indices_random_order)} samples. Plotting {len(indices_to_plot)} random samples.")
        
        class_save_path = os.path.join(save_path, class_name.replace(" ", "_"))
        os.makedirs(class_save_path, exist_ok=True)
        
        # The rest of your logic was already correct for plotting a single sample
        for i, sample_idx in enumerate(indices_to_plot):
            per_modality_llrs = {
                mod: llrs_dict[mod][0][sample_idx] for mod in modalities
            }
            
            summed_llrs = np.sum([np.array(llrs) for llrs in per_modality_llrs.values()], axis=0)
            pred_label_idx = np.argmax(summed_llrs)
            contributions = [per_modality_llrs[mod][pred_label_idx] for mod in modalities]
            magnitudes = np.abs(np.array(contributions))
            contribution_scores = 1 / (magnitudes + 1e-9)

            pred_label_name = idx_to_label.get(pred_label_idx, "N/A")
            title = f"Contributions for '{pred_label_name}' (True: '{class_name}', Random Sample #{i+1})"

            _plot_contribution_chart(
                modalities=modalities,
                scores=contribution_scores,
                colors=colors,
                title=title,
                ylabel="Contribution (1 / |LLR|)",
                save_path=class_save_path,
                filename=f"random_sample_{i+1}"
            )


def calculate_metrics(y_true, y_pred):
    '''
    precision and recall are ill-defined and being set to 0.0 in labels with no predicted samples
    '''
    
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro')

    return precision, recall, f1
