from __future__ import annotations

import numpy as np


DTYPE = np.float32  # 128


def update_step_avgs_and_covs(
    xnew: np.ndarray[DTYPE],
    ynew: np.ndarray[DTYPE],
    n: np.uint32,
    avgx: np.ndarray[DTYPE],
    avgy: np.ndarray[DTYPE],
    covx: np.ndarray[DTYPE],
    covy: np.ndarray[DTYPE],
    covxy: np.ndarray[DTYPE],
) -> tuple[
    np.ndarray[DTYPE],  # avgx
    np.ndarray[DTYPE],  # avgy
    np.ndarray[DTYPE],  # covx
    np.ndarray[DTYPE],  # covy
    np.ndarray[DTYPE],  # covxy
]:

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online

    dx = xnew - avgx
    dy = ynew - avgy

    ncovx = n * covx
    ncovy = n * covy
    ncovxy = n * covxy

    n += 1

    avgx += dx / n
    avgy += dy / n

    ncovx += np.outer(dx, dx)
    ncovy += np.outer(dy, dy)
    ncovxy += np.outer(dy, dx)

    covx = ncovx / n
    covy = ncovy / n
    covxy = ncovxy / n

    return avgx, avgy, covx, covy, covxy


def update_step_for_regression(
    xnew: np.ndarray[DTYPE],
    ynew: np.ndarray[DTYPE],
    n: np.uint32,
    avgx: np.ndarray[DTYPE],
    avgy: np.ndarray[DTYPE],
    varx: np.ndarray[DTYPE],
    covxy: np.ndarray[DTYPE],
) -> tuple[
    np.ndarray[DTYPE],  # avgx
    np.ndarray[DTYPE],  # avgy
    np.ndarray[DTYPE],  # varx
    np.ndarray[DTYPE],  # covxy
]:

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online

    dx = xnew - avgx
    dy = ynew - avgy

    nvarx = n * varx
    ncovxy = n * covxy

    n += 1

    avgx += dx / n
    avgy += dy / n

    nvarx += dx ** 2.0
    ncovxy += np.outer(dy, dx)

    varx = nvarx / n
    covxy = ncovxy / n

    return avgx, avgy, varx, covxy


def stream_avgs_and_covs(
    x: np.ndarray[DTYPE],
    y: np.ndarray[DTYPE],
) -> tuple[
    np.ndarray[DTYPE],  # avgx
    np.ndarray[DTYPE],  # avgy
    np.ndarray[DTYPE],  # covx
    np.ndarray[DTYPE],  # covy
    np.ndarray[DTYPE],  # covxy
]:

    num_x_vars, num_samples = x.shape
    assert y.shape[1] == num_samples
    num_y_vars = y.shape[0]

    for idx_sample in range(num_samples):

        xnew = x[:, idx_sample].astype(DTYPE)
        ynew = y[:, idx_sample].astype(DTYPE)

        if idx_sample == 0:

            # initializations
            avgx = np.zeros(num_x_vars, dtype=DTYPE)
            avgy = np.zeros(num_y_vars, dtype=DTYPE)
            covx = np.zeros((num_x_vars, num_x_vars), dtype=DTYPE)
            covy = np.zeros((num_y_vars, num_y_vars), dtype=DTYPE)
            covxy = np.zeros((num_y_vars, num_x_vars), dtype=DTYPE)

        else:

            nseen = idx_sample
            avgx, avgy, covx, covy, covxy = update_step_avgs_and_covs(
                xnew, ynew, nseen, avgx, avgy, covx, covy, covxy)

    return avgx, avgy, covx, covy, covxy


def stream_linear_regression(
    x: np.ndarray[DTYPE],
    y: np.ndarray[DTYPE],
) -> tuple[
    np.ndarray[DTYPE],  # coeff_matrix
    np.ndarray[DTYPE],  # intercept
]:

    num_x_vars, num_samples = x.shape
    assert y.shape[1] == num_samples
    num_y_vars = y.shape[0]

    for idx_sample in range(num_samples):

        xnew = x[:, idx_sample].astype(DTYPE)
        ynew = y[:, idx_sample].astype(DTYPE)

        if idx_sample == 0:

            # initializations
            avgx = np.zeros(num_x_vars, dtype=DTYPE)
            avgy = np.zeros(num_y_vars, dtype=DTYPE)
            varx = np.zeros(num_x_vars, dtype=DTYPE)
            covxy = np.zeros((num_y_vars, num_x_vars), dtype=DTYPE)

        else:

            nseen = idx_sample
            avgx, avgy, varx, covxy = update_step_for_regression(
                xnew, ynew, nseen, avgx, avgy, varx, covxy)

    # compute coefficient matrix and intercept
    # tomporary reference:
    # https://stats.stackexchange.com/questions/23481/are-there-algorithms-for-computing-running-linear-or-logistic-regression-param

    mask_varx_nonzero = varx != 0.0
    coeff_matrix = np.zeros((num_y_vars, num_x_vars), dtype=DTYPE)

    coeff_matrix[:, mask_varx_nonzero] = \
        covxy[:, mask_varx_nonzero] / varx[mask_varx_nonzero]
    intercept = avgy - coeff_matrix @ avgx

    return coeff_matrix, intercept, avgx, avgy, varx, covxy


def main():
    pass


if __name__ == '__main__':
    main()
