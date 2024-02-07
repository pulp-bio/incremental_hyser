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


MINIBATCH_SIZE_INFER = 8192  # minibatch size for inference


WINDOW = 256
SLIDE = 64

class _EMGPytorchDataset():

    """
    For PyTorch, a "dataset" is just an object with __getitem__ and __len__
    """

    def __init__(
        self,
        x: np.ndarray[np.float32],
        y: np.ndarray[np.float32] | None = None,
    ):

        _, num_samples = x.shape
        if y is not None:
            assert y.shape[1] == num_samples
        
        self.x = x
        self.y = y
        self.num_samples = num_samples
        self.num_windows = (num_samples - WINDOW) // SLIDE + 1 

    def __len__(self) -> int:
        return self.num_windows

    def __data_generation(self, idx_win: int,
    ) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:

        assert idx_win <= self.num_windows

        idx_start = idx_win * SLIDE
        idx_end = idx_start + WINDOW
        if self.y is None:
            return self.x[:, idx_start:idx_end]
        else:
            return self.x[:, idx_start:idx_end], self.y[:, idx_end - 1]

    def __getitem__(self, idx_win: int,
    ) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
        return self.__data_generation(idx_win)


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
class LoaderMode(enum.Enum):
    TRAINING_RANDOMIZED = 'TRAINING_RANDOMIZED'
    TRAINING_SEQUENTIAL = 'TRAINING_SEQUENTIAL'
    INFERENCE = 'INFERENCE'


def _dataset2dataloader(
    dataset: _EMGPytorchDataset,
    loadermode: LoaderMode,
    minibatch_size: int,
) -> torch.utils.data.DataLoader:

    assert isinstance(loadermode, LoaderMode)

    if loadermode is LoaderMode.TRAINING_RANDOMIZED:
        batch_size = minibatch_size
        drop_last = True
        shuffle = True
        sampler = None
    elif loadermode is LoaderMode.TRAINING_SEQUENTIAL:
        batch_size = minibatch_size
        drop_last = True
        shuffle = False
        sampler = torch.utils.data.SequentialSampler(dataset)
    elif loadermode is LoaderMode.INFERENCE:
        batch_size = minibatch_size
        drop_last = False
        shuffle = False
        sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        raise NotImplementedError

    collate_fn = _collate_x_only if dataset.y is None else _collate_xy_pairs

    dataloader = torch.utils.data.DataLoader(
        dataset,  # just an arg, not a kwarg
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
    dataloader = _dataset2dataloader(
        dataset,
        loadermode=LoaderMode.INFERENCE,
        minibatch_size=MINIBATCH_SIZE_INFER,
    )

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
    ytrain: np.ndarray[np.float32],
    xvalid: np.ndarray[np.float32] | None,
    yvalid: np.ndarray[np.float32] | None,
    model: torch.nn.Module,
    loadermode_train: \
        [LoaderMode.TRAINING_RANDOMIZED, LoaderMode.TRAINING_SEQUENTIAL],
    minibatch_train: int,
    # minibatch_valid: int,
    num_epochs: int,
    criterion: torch.nn.Module | None = None,  # None as default (ugly)
    optimizer: torch.optim.Optimizer | None = None,  # None as default (ugly)
    
) -> dict:

    assert (xvalid is None) == (yvalid is None)
    assert loadermode_train in \
        [LoaderMode.TRAINING_RANDOMIZED, LoaderMode.TRAINING_SEQUENTIAL]
    
    dataset_train = _EMGPytorchDataset(xtrain, ytrain)
    dataloader_train = _dataset2dataloader(
        dataset_train, loadermode_train, minibatch_train)

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

    history_dict = {
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

        # an ugly patch for windowing the grout-truths too
        # TODO: implement it better
        ytrain_win = ytrain[:, WINDOW - 1 :: SLIDE]
        yvalid_win = yvalid[:, WINDOW - 1 :: SLIDE]

        yout_train_win = do_inference(xtrain, model)
        metrics_train_epoch = \
            good.compute_regression_metrics(ytrain_win, yout_train_win.T)

        if xvalid is not None and yvalid is not None:
            yout_valid_win = do_inference(xvalid, model)
            metrics_valid_epoch = \
                good.compute_regression_metrics(yvalid_win, yout_valid_win.T)
        else:
            yout_valid_win = None
            metrics_valid_epoch = None

        t_end_epoch_s = time.time()
        deltat_epoch_s = t_end_epoch_s - t_start_epoch_s

        if xvalid is not None and yvalid is not None:
            print("%d/%d\t\t%.4f\t%.4f\t\t%.4f\t%.4f\t\t%.1f" % (
                idx_epoch + 1,
                num_epochs,
                np.sqrt(np.mean(np.square(metrics_train_epoch['rmse']))),
                metrics_train_epoch['mae'].mean(),
                np.sqrt(np.mean(np.square(metrics_valid_epoch['rmse']))),
                metrics_valid_epoch['mae'].mean(),
                deltat_epoch_s,
            ))

        else:
            print("%d/%d\t\t%.4f\t%.4f\t\tnone\tnone\t\t%.1f" % (
                idx_epoch + 1,
                num_epochs,
                np.sqrt(np.mean(np.square(metrics_train_epoch['rmse']))),
                metrics_train_epoch['mae'].mean(),
                deltat_epoch_s,
            ))


        history_dict['epoch'][idx_epoch] = {
            'training': metrics_train_epoch,
            'validation': metrics_valid_epoch,
        }

        labels_dict = {
            'ytrain': ytrain_win,
            'yvalid': ytrain_win,
            'yout_train': yout_train_win,
            'yout_valid': yout_valid_win,
            
        }

        training_summary_dict = {
            'model': model,
            'labels': labels_dict,
            'history': history_dict,
        }

    return training_summary_dict


def main() -> None:
    pass


if __name__ == "__main__":
    main()
