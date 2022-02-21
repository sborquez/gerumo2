"""
Metrics Visualizations
======================

Generate plot for different metrics of models.

Here you can find training metrics, single model evaluation
and models comparison.
"""
from os.path import join
import matplotlib.pyplot as plt


"""
Training Metrics
================
"""


def training_history(history, training_time, model_name,
                     ylog=False, save_to=None):
    """
    Display training loss and validation loss vs epochs.
    """
    fig = plt.figure(figsize=(12, 6))
    epochs = len(history.history['loss'])  # fix: early stop
    epochs = [i for i in range(1, epochs + 1)]
    plt.plot(epochs, history.history['loss'], "--", label="Train")
    plt.plot(epochs, history.history['val_loss'], "--", label="Validation")

    # Style
    title = f'Model {model_name} Training Loss\n '
    title += f'Training time {training_time:.2f} [min]'
    plt.title(title)
    if ylog:
        plt.ylabel('log Loss')
        plt.yscale('log')
    else:
        plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.xticks(rotation=-90)
    plt.grid(True)

    # Show or Save
    if save_to is not None:
        if ylog:
            fig.savefig(join(save_to, f'{model_name} - Training Log Loss.png'))
        else:
            fig.savefig(join(save_to, f'{model_name} - Training Loss.png'))
        plt.close(fig)
    else:
        plt.show()
