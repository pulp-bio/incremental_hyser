from __future__ import annotations
import enum

import numpy as np
import scipy.stats as s
from sklearn import metrics as m


@enum.unique
class RegressionMetric(enum.Enum):
    MAE = 'mae'
    RMSE = 'rmse'
    R2 = 'r2'
    PEARSON = 'pearson'
    SPEARMAN = 'spearman'


def pairwise_correlation(
    x: np.ndarray[np.float32],
    y: np.ndarray[np.float32],
    corr_type: str,
) -> np.ndarray[np.float32]:

    num_vars, num_samples = x.shape
    assert y.shape == (num_vars, num_samples)
    assert num_vars < num_samples
    assert corr_type in ['pearson', 'spearman']

    corrs = np.zeros(num_vars, dtype=np.float32)
    for idx_var in range(num_vars):

        if corr_type == 'pearson':
            corrs[idx_var] = np.corrcoef(x[idx_var], y[idx_var])[0, 1]

        elif corr_type == 'spearman':
            corrs[idx_var] = s.spearmanr(x[idx_var], y[idx_var]).correlation

        else:
            raise NotImplementedError

    return corrs


def compute_regression_metrics(
    ytrue: np.ndarray[np.float32],
    ypred: np.ndarray[np.float32],
) -> dict:

    # Mean Absolute Error (MAE)
    mae = m.mean_absolute_error(ytrue.T, ypred.T, multioutput='raw_values')

    # Root Mean Square Error (RMSE)
    mse = m.mean_squared_error(ytrue.T, ypred.T, multioutput='raw_values')
    rmse = np.sqrt(mse)
    del mse

    # coefficient of determination R2
    r2 = m.r2_score(ytrue.T, ypred.T, multioutput='raw_values')

    # Pearson's correlation coefficient
    pearson = pairwise_correlation(ytrue, ypred, corr_type='pearson')

    # Spearman's rank correlation coefficient
    spearman = pairwise_correlation(ytrue, ypred, corr_type='spearman')

    # store into a dictionary
    regression_metrics = {
        'ytrue': ytrue,
        'ypred': ypred,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson': pearson,
        'spearman': spearman,
    }

    return regression_metrics


def main() -> None:
    pass


if __name__ == '__main__':
    main()
