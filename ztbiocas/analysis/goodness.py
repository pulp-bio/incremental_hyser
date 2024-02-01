from __future__ import annotations
import enum

# non-torch imports
import numpy as np
from sklearn import utils as sklutils
from sklearn import metrics as m
# torch imports
import torch  # just for tensors and datatypes
import torch.nn.functional as F


def balanced_crossentropy_score(
    ytrue: np.ndarray[np.uint8],
    yout: np.ndarray[np.float32],
) -> float:

    """
    Shortcut by wrapping PyTorch to exploit the function.
    https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
    which redirects to
    https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    """

    # compute class weights
    num_classes = yout.shape[1]
    class_labels_array = np.arange(num_classes, dtype=np.uint8)
    class_weights = sklutils.class_weight.compute_class_weight(
        class_weight='balanced', classes=class_labels_array, y=ytrue)

    # convert to torch.Tensor
    # PyTorch's crossentropy wants int64 format for the target labels
    ytrue = torch.tensor(
        ytrue, dtype=torch.int64, requires_grad=False, device='cpu')
    yout = torch.tensor(
        yout, dtype=torch.float32, requires_grad=False, device='cpu')
    class_weights = torch.tensor(
        class_weights, dtype=torch.float32, requires_grad=False, device='cpu')

    # remember that PyTorch passes pred and true swapped wrt to Scikit-Learn
    balanced_crossentropy = F.cross_entropy(yout, ytrue, weight=class_weights)
    balanced_crossentropy = balanced_crossentropy.item()

    return balanced_crossentropy


def compute_classification_metrics(
    ytrue: np.ndarray[np.uint8],
    yout: np.ndarray[np.float32],
) -> dict:

    # compute metrics' values

    yhard = yout.argmax(1)  # yout has shape (num_examples, num_classes)
    yhard = yhard.astype(np.uint8)

    # balanced crossentropy
    balanced_crossentropy = balanced_crossentropy_score(ytrue, yout)

    # balanced accuracy
    balanced_accuracy = m.balanced_accuracy_score(ytrue, yhard)

    # accuracy
    accuracy = m.accuracy_score(ytrue, yhard)

    # store into a dictionary

    metrics_dict = {
        'balanced_crossentropy': balanced_crossentropy,
        'balanced_accuracy': balanced_accuracy,
        'accuracy': accuracy,
    }

    return metrics_dict


@enum.unique
class MLTask(enum.Enum):
    CLASSIFICATION = 'CLASSIFICATION'
    REGRESSION = 'REGRESSION'


def compute_regression_metrics(
    ytrue: np.ndarray[np.float32],
    yout: np.ndarray[np.float32],
) -> dict:

    # compute metrics' values

    # RMSE
    rmse = np.sqrt(m.mean_squared_error(ytrue, yout))

    # MAE
    mae = m.mean_absolute_error(ytrue, yout)

    # store into a dictionary

    metrics_dict = {
        'rmse': rmse,
        'rmse': mae,
    }

    return metrics_dict


def compute_metrics(
    ytrue: np.ndarray[np.float32],
    yout: np.ndarray[np.float32],
    mltask: MLTask,
) -> dict:
    
    assert mltask in MLTask

    if mltask is MLTask.CLASSIFICATION:
        metrics_dict = compute_classification_metrics(ytrue, yout)
    elif mltask is MLTask.REGRESSION:
        metrics_dict = compute_regression_metrics(ytrue, yout)
    else:
        raise NotImplementedError

    return metrics_dict


def main() -> None:
    pass


if __name__ == '__main__':
    main()
