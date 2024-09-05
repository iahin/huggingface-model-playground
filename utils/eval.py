import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
import seaborn as sns


def f1metrics(true_labels, predicted_labels):
    # Calculate precision, recall, and F1 score
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(
        true_labels, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(
        true_labels, predicted_labels, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print(
        f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")


def plot_loss_graph(train_losses, val_losses, save_path=None):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, cmap=plt.cm.Blues, save_path=None):
    """
    Plot a confusion matrix.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        classes (list): Class labels.
        normalize (bool): Whether to normalize the values.
        cmap (matplotlib colormap): Colormap for the plot.
        save_path: path to save plots
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
    else:
        title = 'Confusion Matrix'

    # Only use the labels that appear in the data
    classes = [classes[i] for i in unique_labels(y_true, y_pred)]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_data_distribution(df, class_column, save_path=None):
    """
    Plot the data distribution based on the specified text and class columns.

    Parameters:
        df (pandas DataFrame): The DataFrame containing the data.
        class_column (str): The column name for the class labels.
    """
    plt.figure()
    ax = sns.countplot(x=class_column, data=df)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Data Distribution')

    # Annotate each bar with its count
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def multi_model_fscore(df, true_label_column, save_path=None):
    # Calculate precision, recall, and F1 score for each model
    models = df.columns[1:]
    precision_scores = [precision_score(df[true_label_column], df[model], average='weighted', zero_division=1) for model in models]
    recall_scores = [recall_score(df[true_label_column], df[model], average='weighted') for model in models]
    f1_scores = [f1_score(df[true_label_column], df[model], average='weighted') for model in models]

    # Plot the scores
    fig, ax = plt.subplots(figsize=(10, 5))

    bar_width = 0.2
    bar_positions = range(len(models))

    ax.bar(bar_positions, precision_scores, width=bar_width, label='Precision')
    ax.bar([pos + bar_width for pos in bar_positions], recall_scores, width=bar_width, label='Recall')
    ax.bar([pos + 2 * bar_width for pos in bar_positions], f1_scores, width=bar_width, label='F1 Score')

    for pos, value in zip(bar_positions, precision_scores):
        ax.text(pos, value, f'{value:.2f}', ha='center', va='bottom')

    for pos, value in zip([pos + bar_width for pos in bar_positions], recall_scores):
        ax.text(pos, value, f'{value:.2f}', ha='center', va='bottom')

    for pos, value in zip([pos + 2 * bar_width for pos in bar_positions], f1_scores):
        ax.text(pos, value, f'{value:.2f}', ha='center', va='bottom')

    ax.set_xticks([pos + bar_width for pos in bar_positions])
    ax.set_xticklabels(models)
    ax.set_xlabel('Model Names')
    ax.set_ylabel('Score')
    ax.legend()

    plt.title('Precision, Recall, and F1 Score for Multiple Models')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
