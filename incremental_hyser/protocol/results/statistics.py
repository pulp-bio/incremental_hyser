from __future__ import annotations
import pickle

import numpy as np
import scipy.stats as sstt

from ...hyser import hyser as hy
from ...regression import goodness as good


def load_results_dict(
    results_dict_src_filepath: str,
) -> dict:

    with open(results_dict_src_filepath, 'rb') as f:
        results_dict = pickle.load(f)
        results_dict = results_dict['results']  # peel outer key

    return results_dict


def take_val_stats_single_metric(
    results_dict: dict,
    metric: good.RegressionMetric,
    stage: str,
) -> dict:

    """
    Compute the statistics of a single regression metric on the validation set.
    """

    # Statistics
    # - median, Inter-Quartile Range (IQR), and Median Absolute Deviation (MAD)
    # - average and standard deviation

    assert stage in ['training', 'validation']

    # skip everything if the metric is not in every subject
    for idx_subj in range(hy.NUM_SUBJECTS):
        if metric.value not in results_dict['subject'][idx_subj][stage].keys():
            return

    # collect all values into an array
    metric_vals = np.array(
        [
            results_dict['subject'][_][stage][metric.value]
            for _ in range(hy.NUM_SUBJECTS)
        ],
        dtype=np.float32,
    )

    # TODO: DECIDE THE FOLLOWING!

    # First aggregate over fingers, than over subjects
    metric_med = np.median(np.median(metric_vals, 1))
    metric_mad = sstt.median_abs_deviation(np.median(metric_vals, 1))
    metric_iqr = sstt.iqr(np.median(metric_vals, 1))
    metric_avg = metric_vals.mean(1).mean()
    metric_std = metric_vals.mean(1).std()

    # Aggregate over both fingers and subjects, at the same level
    # metric_med = np.median(metric_vals)
    # metric_mad = sstt.median_abs_deviation(metric_vals, axis=None)
    # metric_iqr = sstt.iqr(metric_vals)
    # metric_avg = metric_vals.mean()
    # metric_std = metric_vals.std()

    # display validation results
    # if stage == 'validation':
    #    print(
    #        f"{metric.value}\n"
    #        f"\n"
    #        f"median +/- iqr\t\tmedian +/- mad\t\tavg +/- std\n"
    #        f"{metric_med:.4f} +/- {metric_iqr:.4f}\t"
    #        f"{metric_med:.4f} +/- {metric_mad:.4f}\t"
    #        f"{metric_avg:.4f} +/- {metric_std:.4f}\n"
    #    )

    # structure into a dictionary
    stats_dict = {
        'med': metric_med,
        'iqr': metric_iqr,
        'mad': metric_mad,
        'avg': metric_avg,
        'std': metric_std,
    }

    return stats_dict


def take_stats_of_all_metrics(
    results_dict: dict,
) -> dict:

    """
    Compute the statistics of all regression metrics on the validation set.
    """

    # print(
    #    f"\n----------------------------------------------------------------\n"
    #    f"\n"
    #    f"VALIDATION RESULTS\n"
    #    f"\n"
    # )

    stats_dict_allmetrics = {
        'training': {},
        'validation': {},
    }

    for stage in stats_dict_allmetrics.keys():
        for metric in good.RegressionMetric:
            stats_dict_allmetrics[stage][metric.value] = \
                take_val_stats_single_metric(results_dict, metric, stage)

    # print(
    #    f"\n----------------------------------------------------------------\n"
    # )

    return stats_dict_allmetrics


def inspect_results(
    results_dict_src_filepath: str,
) -> dict:

    results_dict = load_results_dict(results_dict_src_filepath)

    stats_dict_allmetrics = take_stats_of_all_metrics(results_dict)
    # make_plots_all_subjects(results_dict)

    return results_dict, stats_dict_allmetrics


def main() -> None:
    pass


if __name__ == '__main__':
    main()
