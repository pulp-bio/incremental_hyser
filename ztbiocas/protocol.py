from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch.nn  # just for nn.Module

from online import pca as opca
from learning.learning import do_inference


def calibration_experiment(
    xcalib: np.ndarray[np.float32],
    ycalib: np.ndarray,  # dtype of classification or regression
    xvalid: np.ndarray[np.float32],
    yvalid: np.ndarray,  # dtype of classification or regression
    adapt_flag: bool,
    beta: float,
    stdscaler_train: StandardScaler,
    pca_train: PCA | None,
    model: torch.nn.Module,
    output_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:  # dtype of classification or regression

    num_channels = xcalib.shape[0]

    xcalib_std = stdscaler_train.transform(xcalib.T).T
    xvalid_std = stdscaler_train.transform(xvalid.T).T
    del xcalib, xvalid

    if adapt_flag:

        # online PCA

        W_init = opca.initialize_online_pca(
            num_channels, opca.InitMode.CUSTOM, pca_train.components_.T)
        num_samples_calib = xcalib_std.shape[1]
        ids_samples = np.arange(num_samples_calib)
        
        # scheduling
        gamma_scheduled = 1.0 / (1.0 + ids_samples / beta)
        
        W_sequence, mean_calib, scale_calib = opca.oja_sga_session(
            xcalib_std, W_init, gamma_scheduled)
        W_calib = W_sequence[-1]
        W_calib = opca.reorder_w_like_reference_pca(
            W_calib, pca_train.components_)

        mean_calib = np.expand_dims(mean_calib, axis=1)
        scale_calib = np.expand_dims(scale_calib, axis=1)

        xcalib_std = (xcalib_std - mean_calib) / scale_calib
        xvalid_std = (xvalid_std - mean_calib) / scale_calib
        
        # apply PCA

        xcalib_pc = W_calib.T @ xcalib_std
        xvalid_pc = W_calib.T @ xvalid_std

    else:
        # no adptation
        xcalib_pc = pca_train.transform(xcalib_std.T).T
        xvalid_pc = pca_train.transform(xvalid_std.T).T

    del xcalib_std, xvalid_std

    # inference
    yout_calib = do_inference(xcalib_pc, model, output_scale)
    yout_valid = do_inference(xvalid_pc, model, output_scale)
    del xcalib_pc, xvalid_pc

    return yout_calib, yout_valid


def main() -> None:
    pass


if __name__ == "__main__":
    main()
