from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch.nn  # just for nn.Module

from .online import pca as opca
from .learning.learning import do_inference
from .analysis.goodness import compute_classification_metrics
# to be generalized



def online_pca_calibration_session(
    xcalib_std: np.ndarray[np.float32],
    w_init: np.ndarray[np.float32],
    beta: float,
) -> tuple[
    np.ndarray[np.float32],  # mean_calib
    np.ndarray[np.float32],  # std_calib
    np.ndarray[np.float32],  # w_calib
]:

    num_ch, num_samples = xcalib_std.shape
    ids_samples = np.arange(num_samples)
    gamma_scheduled = 1.0 / (1.0 + ids_samples / beta)

    w_sequence, mean_calib, scale_calib = \
        opca.oja_sga_session(xcalib_std, w_init, gamma_scheduled)
    w_calib = w_sequence[-1]

    # heuristic reordering
    # (BioCAS code used the NON-transposed pca_train.components_)
    w_calib = opca.reorder_w_like_reference_pca(w_calib, w_init.T)

    mean_calib = np.expand_dims(mean_calib, axis=1)  # equiv. to online
    scale_calib = np.expand_dims(scale_calib, axis=1)  # equiv. to online

    return mean_calib, scale_calib, w_calib



def calibration_experiment(
    xcalib: np.ndarray[np.float32],
    ycalib,
    xvalid: np.ndarray[np.float32],
    yvalid,
    beta: float,
    stdscaler_train: StandardScaler,
    pca_train: PCA,
    model: torch.nn.Module,
    output_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:  # dtype of classification or regression

    # standard scaling, always frozen
    xcalib_std = stdscaler_train.transform(xcalib.T).T
    xvalid_std = stdscaler_train.transform(xvalid.T).T
    del xcalib, xvalid

    # PCA
    # frozen
    xcalib_pc_froz = pca_train.transform(xcalib_std.T).T
    xvalid_pc_froz = pca_train.transform(xvalid_std.T).T
    # refit    
    # ----------------------------------------------------------------------- #
    w_init = pca_train.components_.T
    mean_calib, scale_calib, w_calib = \
        online_pca_calibration_session(xcalib_std, w_init, beta)
    xcalib_pc_refit = w_calib.T @ (xcalib_std - mean_calib) / scale_calib
    xvalid_pc_refit = w_calib.T @ (xvalid_std - mean_calib) / scale_calib
    # ----------------------------------------------------------------------- #

    del xcalib_std, xvalid_std

    # inference
    # frozen
    yout_calib_froz = do_inference(xcalib_pc_froz, model, output_scale)
    yout_valid_froz = do_inference(xvalid_pc_froz, model, output_scale)
    del xcalib_pc_froz, xvalid_pc_froz
    # refit
    yout_calib_refit = do_inference(xcalib_pc_refit, model, output_scale)
    yout_valid_refit = do_inference(xvalid_pc_refit, model, output_scale)
    del xcalib_pc_refit, xvalid_pc_refit

    # yout_dict = {
    #    'calibration': {
    #        'frozen': yout_calib_froz,
    #        'refit': yout_calib_refit,
    #    },
    #    'validation': {
    #        'frozen': yout_valid_froz,
    #        'refit': yout_valid_refit,
    #    },
    # }

    metrics_calib_froz = compute_classification_metrics(ycalib, yout_calib_froz)
    metrics_valid_froz = compute_classification_metrics(yvalid, yout_valid_froz)
    metrics_calib_refit = compute_classification_metrics(ycalib, yout_calib_refit)
    metrics_valid_refit = compute_classification_metrics(yvalid, yout_valid_refit)
    metrics = {
        'calibration': {
            'frozen': metrics_calib_froz,
            'refit': metrics_calib_refit,
        },
        'validation': {
            'frozen': metrics_valid_froz,
            'refit': metrics_valid_refit,
        },
    }

    return metrics


def main() -> None:
    pass


if __name__ == "__main__":
    main()
