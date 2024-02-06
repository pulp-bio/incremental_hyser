from __future__ import annotations
import enum
import time

# non-torch imports
import numpy as np
import sklearn.utils as sklutils
# torch imports
import torch
import torch.utils.data

from .settings import DEVICE
from . import goodness as good


MINIBATCH_SIZE_TRAIN = 16
MINIBATCH_SIZE_INFER = 8192  # minibatch size for inference


class _EMGPytorchDataset():

    """
    For PyTorch, a "dataset" is just an object with __getitem__ and __len__
    """

    def __init__(
        self,
        x: np.ndarray[np.float32],
        y: np.ndarray[np.float32] | None = None,
    ):

        # "examples" is less ambiguous than "samples": not the single numbers
        _, num_examples = x.shape
        if y is not None:
            assert len(y) == num_examples
        
        self.x = x
        self.y = y
        self.num_examples = num_examples

    def __len__(self) -> int:
        return self.num_examples

    def __data_generation(self, idx_example: int,
    ) -> tuple[np.ndarray[np.float32], np.ndarray[np.uint8 | np.float32]]:

        # the format is (num_ch, num_exsamples): indicize the second dimension
        assert idx_example <= self.num_examples
        if self.y is None:
            return self.x[:, idx_example]
        else:
            return self.x[:, idx_example], self.y[idx_example]

    def __getitem__(self, idx: int,
    ) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
        return self.__data_generation(idx)


def _collate_x_only(
    minibatch: list[np.ndarray[np.float32]]
) -> torch.Tensor[torch.float32]:

    # concatenating in NumPy first should be faster
    x = np.array(minibatch, dtype=np.float32)
    del minibatch
    x = torch.tensor(x, dtype=torch.float32, requires_grad=False, device='cpu')

    return x


def _collate_xy_pairs(
    minibatch: list[tuple[np.ndarray[np.float32], np.ndarray[np.float32]]]
) -> tuple[torch.Tensor[torch.float32], torch.Tensor[torch.float32]]:

    # concatenating in NumPy first should be faster
    x = np.array([xy[0] for xy in minibatch], dtype=np.float32)
    y = np.array([xy[1] for xy in minibatch], dtype=np.float32)
    del minibatch

    x = torch.tensor(x, dtype=torch.float32, requires_grad=False, device='cpu')
    # PyTorch's crossentropy wants int64 format for the target labels
    # here, do not specify the type to use it for classification and regression
    y = torch.tensor(y, requires_grad=False, device='cpu')

    return x, y


@enum.unique
class _Mode(enum.Enum):
    TRAINING = 'TRAINING'
    INFERENCE = 'INFERENCE'


def _dataset2dataloader(
    dataset: _EMGPytorchDataset,
    mode: _Mode,
) -> torch.utils.data.DataLoader:

    assert isinstance(mode, _Mode)

    if mode == _Mode.TRAINING:
        batch_size = MINIBATCH_SIZE_TRAIN
        drop_last = True
        shuffle = True
        sampler = None
    elif mode == _Mode.INFERENCE:
        batch_size = MINIBATCH_SIZE_INFER
        drop_last = False
        shuffle = False
        sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        raise ValueError

    collate_fn = _collate_x_only if dataset.y is None else _collate_xy_pairs

    dataloader = torch.utils.data.DataLoader(
        dataset,  # just arg, not kwarg
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    return dataloader


def do_inference(
    x: np.ndarray[np.float32],
    model: torch.nn.Module,
    output_scale: float = 1.0,
) -> tuple:

    dataset = _EMGPytorchDataset(x)
    dataloader = _dataset2dataloader(dataset, mode=_Mode.INFERENCE)

    model.eval()
    model.to(DEVICE)
    
    first_minibatch = True
    for x_b in dataloader:
        x_b = x_b.to(DEVICE)
        with torch.no_grad():
            yout_b = model(x_b)
        del x_b
        yout_b = yout_b.detach()
        yout_b = yout_b.cpu()
        yout_b = yout_b.numpy()
        if first_minibatch:
            yout = yout_b
            first_minibatch = False
        else:
            yout = np.concatenate((yout, yout_b), axis=0)
        del yout_b

    yout *= output_scale

    return yout


def do_training(
    xtrain: np.ndarray[np.float32],
    ytrain: np.ndarray[np.uint8 | np.float32],
    xvalid: np.ndarray[np.float32] | None,
    yvalid: np.ndarray[np.uint8 | np.float32] | None,
    model: torch.nn.Module,
    minibatch_train: int,
    minibatch_valid: int,
    num_epochs: int,
    criterion: torch.nn.Module | None = None,  # None as default (ugly)
    optimizer: torch.optim.Optimizer | None = None,  # None as default (ugly)
    
) -> dict:

    assert (xvalid is None) == (yvalid is None)
    
    dataset_train = _EMGPytorchDataset(xtrain, ytrain)
    dataloader_train = _dataset2dataloader(dataset_train, mode=_Mode.TRAINING)

    model.to(DEVICE)
    model.train()

    if criterion is None:
        criterion = torch.nn.MSELoss()
    criterion.to(DEVICE)

    if optimizer is None:
        params = model.parameters()
        lr = 0.001
        weight_decay = 0.001
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    history = {
        'epoch': {},
    }

    print(
        f"\n"
        f"\t\tTRAINING\t\tVALIDATION\n"
        f"\n"
        "EPOCH\t\tRMSE\tMAE\t\tRMSE\tMAE\t\tTime (s)\n"
    )
    for idx_epoch in range(num_epochs):

        t_start_epoch_s = time.time()

        for x_b, y_b in dataloader_train:
            x_b = x_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            optimizer.zero_grad()
            yout_b = model(x_b)
            loss_b = criterion(yout_b, y_b)
            loss_b.backward()
            optimizer.step()

        yout_train = do_inference(xtrain, model)
        metrics_train_epoch = \
            good.compute_regression_metrics(ytrain, yout_train)

        if xvalid is not None and yvalid is not None:
            yout_valid = do_inference(xvalid, model)
            metrics_valid_epoch = \
                good.compute_regression_metrics(yvalid, yout_valid)
        else:
            yout_valid = None
            metrics_valid_epoch = None

        t_end_epoch_s = time.time()
        deltat_epoch_s = t_end_epoch_s - t_start_epoch_s

        if xvalid is not None and yvalid is not None:
            print("%d/%d\t\t%.4f\t%.4f\t\t%.4f\t%.4f\t\t%.1f" % (
                idx_epoch + 1,
                num_epochs,
                metrics_train_epoch['rmse'],
                metrics_train_epoch['mae'],
                metrics_valid_epoch['rmse'],
                metrics_valid_epoch['mae'],
                deltat_epoch_s,
            ))

        else:
            print("%d/%d\t\t%.4f\t%.4f\t\tnone\tnone\t\t%.1f" % (
                idx_epoch + 1,
                num_epochs,
                metrics_train_epoch['rmse'],
                metrics_train_epoch['mae'],
                deltat_epoch_s,
            ))


        history['epoch'][idx_epoch] = {
            'training': metrics_train_epoch,
            'validation': metrics_valid_epoch,
        }

        training_summary_dict = {
            'model': model,
            'ytrain': ytrain,
            'yvalid': ytrain,
            'yout_train': yout_train,
            'yout_valid': yout_valid,
            'history': history,
        }

    return training_summary_dict


def main() -> None:
    pass


if __name__ == "__main__":
    main()
