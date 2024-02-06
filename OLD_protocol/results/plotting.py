from __future__ import annotations
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from ...hyser import hyser as hy
from hdsemg_force_regression.regression import goodness as good


"""
def make_plots_single_subject_NOT_AS_PAPER(
    results_dict_allsubj: dict,
    idx_subject: int,  # for labelling
) -> None:

    assert idx_subject in range(hy.NUM_SUBJECTS)
    results_dict_onesubj = results_dict_allsubj['subject'][idx_subject]
    del results_dict_allsubj

    # training
    force_mvc_train = results_dict_onesubj['training']['ytrue']
    force_pred_mvc_train = results_dict_onesubj['training']['ypred']
    # validation
    force_mvc_valid = results_dict_onesubj['validation']['ytrue']
    force_pred_mvc_valid = results_dict_onesubj['validation']['ypred']

    # time
    num_samples_train = force_mvc_train.shape[1]
    num_samples_valid = force_mvc_valid.shape[1]
    time_s_train = np.arange(num_samples_train) / hy.FS_FORCE
    time_s_valid = np.arange(num_samples_valid) / hy.FS_FORCE
    time_max_s_train = num_samples_train / hy.FS_FORCE  # "rounded" by +1
    time_max_s_valid = num_samples_valid / hy.FS_FORCE  # "rounded" by +1

    for idx_finger in range(hy.NUM_FINGERS):

        # training plot

        plt.figure(figsize=(20.0, 5.0))
        plt.title(
            f"SUBJECT {idx_subject + 1}/{hy.NUM_SUBJECTS}\n"
            f"TRAINING SESSION\n"
            f"Finger {idx_finger + 1}/{hy.NUM_FINGERS}"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Force (fraction of MVC)")
        plt.plot(
            time_s_train, force_mvc_train[idx_finger],
            'k-', linewidth=0.5, label="ground truth",
        )
        plt.plot(
            time_s_train, force_pred_mvc_train[idx_finger],
            color='r',
            # color=(0.5, 0.5, 0.5),
            linewidth=0.5, label="regression estimate",
        )
        plt.legend()
        plt.grid()
        plt.axis([0.0, time_max_s_train, -0.75, +0.75])
        plt.show()


        # validation plot

        plt.figure(figsize=(20.0, 5.0))
        plt.title(
            f"SUBJECT {idx_subject + 1}/{hy.NUM_SUBJECTS}\n"
            f"VALIDATION SESSION\n"
            f"Finger {idx_finger + 1}/{hy.NUM_FINGERS}"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Force (fraction of MVC)")
        plt.plot(
            time_s_valid, force_mvc_valid[idx_finger],
            'k-', linewidth=0.5, label="ground truth",
        )
        plt.plot(
            time_s_valid, force_pred_mvc_valid[idx_finger],
            color='r', linewidth=0.5, label="regression estimate",
        )
        plt.legend()
        plt.grid()
        #plt.axis([0.0, time_max_s_valid, -0.75, +0.75])
        plt.axis([0.0, time_max_s_valid, -0.75, +0.75])
        plt.show()

    return
"""


def plot_as_per_paper(
    results_dict_allsubj: dict,
    idx_subject: int,
    trials_to_plot: list,
    fingers_to_plot: list,
    save: bool,
) -> None:
    # only validation plot

    assert idx_subject in range(hy.NUM_SUBJECTS)
    results_dict_onesubj = results_dict_allsubj["subject"][idx_subject]
    del results_dict_allsubj

    force_mvc = results_dict_onesubj["validation"]["ytrue"]
    force_pred_mvc = results_dict_onesubj["validation"]["ypred"]

    # time
    num_samples = hy.NUM_SAMPLES_FORCE_RANDOM
    time_s = np.arange(num_samples) / hy.FS_FORCE
    time_max_s = num_samples / hy.FS_FORCE  # "rounded" by +1

    num_trials_plotted = len(trials_to_plot)
    num_fingers_plotted = len(fingers_to_plot)
    idx_central_trial = num_trials_plotted // 2 + \
        num_trials_plotted % 2 - 1

    finger_name_labels_dict = {
        0: "Thumb",
        1: "Index",
        2: "Middle",
        3: "Ring",
        4: "Little",
    }

    approx_plot_width = 5.0
    approx_plot_height = 4.0
    figsize = (
        approx_plot_width * num_trials_plotted,
        approx_plot_height * num_fingers_plotted,
    )
    fig, axs = plt.subplots(
        figsize=figsize,
        nrows=num_fingers_plotted,
        ncols=num_trials_plotted,
        sharex=True,
        sharey=True,
    )
    # fig.suptitle(f"SUBJECT {idx_subject + 1 : d}")

    for idx_fp in range(num_fingers_plotted):  # "fp": "finger to plot"
        for idx_tp in range(num_trials_plotted):  # "tp": "trial to plot"
            ax = axs[idx_fp, idx_tp]
            idx_trial = trials_to_plot[idx_tp]
            idx_finger = fingers_to_plot[idx_fp]

            # ---------------- #

            # title

            title_str = ""

            if idx_fp == 0:
                title_str += f"Trial {idx_trial + 1 : d}\n "
            if idx_tp == idx_central_trial:
                title_str += f"\nFinger {idx_finger + 1 : d}: {finger_name_labels_dict[idx_finger]:s}"
            else:
                title_str += f"\n"
            
            ax.set_title(title_str, fontsize=16)
            
            # ---------------- #

            # labels

            if idx_fp == num_fingers_plotted - 1:
                ax.set_xlabel("Time (s)", fontsize=14)

            if idx_tp == 0:
                ax.set_ylabel("Force (MVC)", fontsize=14)
            
            # ---------------- #

            # data

            idx_start = idx_trial * hy.NUM_SAMPLES_FORCE_RANDOM
            idx_stop = idx_start + hy.NUM_SAMPLES_FORCE_RANDOM
            ax.plot(
                time_s,
                force_mvc[idx_finger, idx_start:idx_stop],
                linestyle='-',
                color='k',
                linewidth=0.5,
                label="ground truth",
            )
            ax.plot(
                time_s,
                force_pred_mvc[idx_finger, idx_start:idx_stop],
                linestyle='-',
                color="m",
                linewidth=0.5,
                label="regression estimate",
            )

            # ---------------- #

            # adjustments

            ax.grid()
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_xlim([0.0, time_max_s])
            ax.set_ylim([-0.8, +0.8])
            if idx_fp == 0 and idx_tp == 0:
                ax.legend(loc='upper left', fontsize=12)

            # ---------------- #

            del ax

    if save:
        plt.savefig(
            "./plot_forces.pdf", dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.show()

    
    plt.figure(figsize=(10.0, 10.0))
    plt.plot(force_mvc[1], force_pred_mvc[1], '.', markersize=1.0)
    plt.grid()
    plt.axis([-0.8, +0.8, -0.8, +0.8])
    plt.show()

    plt.figure(figsize=(10.0, 10.0))
    plt.plot(force_mvc[2], force_pred_mvc[2], '.', markersize=1.0)
    plt.grid()
    plt.axis([-0.8, +0.8, -0.8, +0.8])
    plt.show()

    return


"""
def make_plots_all_subjects(
    results_dict: dict,
) -> None:

    for idx_subject in range(hy.NUM_SUBJECTS):

        print(
            f"\n------------------------------------------------------------\n"
            f"SUBJECT {idx_subject + 1}/{hy.NUM_SUBJECTS}"
            f"\n------------------------------------------------------------\n"
        )

        make_plots_single_subject(results_dict, idx_subject)

    return
"""


def show_hist_of_alphas(
    resdicts_list: list[dict],
    resdicts_lbl_list: list[str],
) -> np.ndarray[np.float32]:
    num_resdicts = len(resdicts_list)
    assert len(resdicts_lbl_list) == num_resdicts

    # collect alphas

    alphas = np.zeros((num_resdicts, hy.NUM_SUBJECTS, hy.NUM_FINGERS), dtype=np.float32)

    for idx_resdict, idx_subj, idx_fing in product(
        range(num_resdicts),
        range(hy.NUM_SUBJECTS),
        range(hy.NUM_FINGERS),
    ):
        resdict = resdicts_list[idx_resdict]
        alphas[idx_resdict, idx_subj] = (
            resdict["subject"][idx_subj]["regressor"]
            .univariate_regressors[idx_fing]
            .alpha_
        )

    # histogram

    plt.figure(figsize=(4.0, 3.0))
    plt.title(r"Distribution of $\alpha$'s of L1")
    plt.xlabel(r"Log$_{10}\alpha$ (dimensionless)")
    plt.ylabel("Empirical distribution counts\n(dimensionless natural)")

    for idx_resdict in range(num_resdicts):
        x = np.log10(alphas[idx_resdict]).flatten()
        label = resdicts_lbl_list[idx_resdict]
        plt.hist(x=x, bins=None, density=False, alpha=0.5, label=label)

    plt.legend()
    plt.grid()
    plt.axis([-4.0, 0.0, 0.0, 30.0])
    plt.show()

    return alphas


def show_hist_of_metric(
    resdicts_list: list[dict],
    metric: good.RegressionMetric,
    resdicts_lbl_list: list[str],
) -> np.ndarray[np.float32]:
    assert isinstance(metric, good.RegressionMetric)
    num_resdicts = len(resdicts_list)
    assert len(resdicts_lbl_list) == num_resdicts

    # collect alphas

    mvals = np.zeros(  # metric's values
        (num_resdicts, hy.NUM_SUBJECTS, hy.NUM_FINGERS), dtype=np.float32
    )

    for idx_resdict, idx_subj in product(
        range(num_resdicts),
        range(hy.NUM_SUBJECTS),
    ):
        resdict = resdicts_list[idx_resdict]
        mvals[idx_resdict, idx_subj] = resdict["subject"][idx_subj]["validation"][
            metric.value
        ]

    # histogram

    plt.figure(figsize=(4.0, 3.0))
    plt.title(f"Distribution of values of:\n{metric.value}")
    plt.xlabel(metric.value)
    plt.ylabel("Empirical distribution counts\n(dimensionless natural)")

    for idx_resdict in range(num_resdicts):
        x = mvals[idx_resdict].flatten()
        label = resdicts_lbl_list[idx_resdict]
        plt.hist(x=x, bins=None, density=False, alpha=0.5, label=label)

    plt.legend()
    plt.grid()
    plt.axis([0.0, None, 0.0, 120.0])
    # plt.axis([-2.0, 0.0, 0.0, 120.0])
    plt.show()

    return mvals


def main() -> None:
    pass


if __name__ == "__main__":
    main()
