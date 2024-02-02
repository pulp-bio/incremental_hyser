from __future__ import annotations

import numpy as np


def update_cov(
    x_newsample: np.ndarray[np.float32],
    avg: np.ndarray[np.float32],
    cov: np.ndarray[np.float32],
    n_seen: int,
) -> np.ndarray[np.float32]:

    dx = x_newsample - avg
    avg += dx / n_seen

    cov += dx.reshape((-1, 1)) @ dx.reshape((1, -1))  # / (n_seen + 1)

    n_seen += 1

    return avg, cov, n_seen


def cov_online(x: np.ndarray[np.float32]) -> np.ndarray[np.float32]:

    num_ch, num_samples = x.shape

    avg = x[:, 0]
    ncov = np.zeros((num_ch, num_ch), dtype=np.float32)
    n_seen = 1

    for idx_sample in range(1, num_samples):
        x_newsample = x[:, idx_sample]
        avg, ncov, n_seen = update_cov(x_newsample, avg, ncov, n_seen)

    cov = ncov / n_seen

    return avg, cov, n_seen


def main() -> None:
    pass


if __name__ == '__main__':
    main()
