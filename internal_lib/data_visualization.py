import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(y_true: np.array, y_pred: np.array, savefig_path: str = None):
    """
    Function that plot and, eventually, save the confusion matrix
    :param y_true: np.array of true labels
    :param y_pred: np.array of predicted labels
    :param savefig_path: str -> path where the figure will be saved
    :return: None
    """

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

    if savefig_path is not None:
        plt.savefig(savefig_path)
