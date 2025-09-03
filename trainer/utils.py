import mlflow
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


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


def plot_confusion_matrix(cm, skip_null_class):
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
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    plt.close()

def plot_confusion_matrix_percentage(cm, skip_null_class):
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
            ax.text(j, i, f"{val:.1f}%", ha='center', va='center', color='black', fontsize=8)

    plt.title('Confusion Matrix (Percentages)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_percentage.png')  
    mlflow.log_artifact('confusion_matrix_percentage.png')  
    plt.close()
