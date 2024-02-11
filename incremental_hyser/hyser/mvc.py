from __future__ import annotations
from itertools import product
import time

import numpy as np
import matplotlib.pyplot as plt

from . import hyser as hy


MVC_V_DICT = None


def extract_mvc(
    force: np.ndarray[np.float32],
    force_direction: hy.ForceDirection,
) -> float:

    # Force signal and MVC are in any units.

    # As per the HYSER dataset's original paper, with a correction, i.e.
    # making sure the contraction in the opposite direction is never included
    # in the statistics; this is done by inverting the sign if needed, instead
    # of taking the absolite value.

    # Source:
    # https://www.physionet.org/content/hd-semg/1.0.0/toolbox/function/get_mvc.m

    NUM_MAXRANK_SELECTED = 200  # as per original HYSER paper & toolbox

    assert force.shape == (hy.NUM_SAMPLES_FORCE_MVC,)
    assert isinstance(force_direction, hy.ForceDirection)

    if force_direction == hy.ForceDirection.FLEXION:
        force = - force

    force_sorted = np.sort(force)  # sort in ascending order, NOT in-place
    force_maxrank = force_sorted[- NUM_MAXRANK_SELECTED:]
    mvc = force_maxrank.mean()

    return mvc


def init_all_mvcs_dict(
    mvc_init_value: float,
) -> dict:

    # Explicit loops are more readable than nested dictionary comprehensions.
    # Fingers are represented by each element of the innermost array.

    mvc_dict = {'subject': {}}

    for idx_subj in range(hy.NUM_SUBJECTS):

        mvc_dict['subject'][idx_subj] = {'session': {}}

        for idx_sess in range(hy.NUM_SESSIONS):

            mvc_dict['subject'][idx_subj]['session'][idx_sess] = \
                {'direction': {}}

            for fd in hy.ForceDirection:

                mvc_dict['subject'][idx_subj]['session'][idx_sess][
                    'direction'][fd.value] = \
                    mvc_init_value * np.ones(hy.NUM_FINGERS, dtype=np.float32)

    return mvc_dict


def show_mvc_plot(
    force_allfingers_v: np.ndarray[np.float32],
    force_direction: hy.ForceDirection,
    mvc_v: float,
) -> None:

    # (this visualization can be improved)

    times_s = np.arange(hy.NUM_SAMPLES_FORCE_MVC) / hy.FS_FORCE  # times in s

    plt.figure(figsize=(10.0, 5.0))
    plt.title(
        f"{force_direction.value}\n"
        f"|MVC| = {mvc_v :.3f} V"
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Force [V]")

    label_mvc = f"|MVC| = {mvc_v :.3f} V"
    plt.axhline(+ mvc_v, linestyle='--', color='k', label=label_mvc)
    plt.axhline(- mvc_v, linestyle='--', color='k', label=None)

    for idx_finger in range(hy.NUM_FINGERS):
        label = f"finger {idx_finger + 1}"
        plt.plot(times_s, force_allfingers_v[idx_finger], '-', label=label)

    plt.legend(loc='center right')
    plt.axis([0.0, hy.TIME_SEGM_MVC_S * 1.2, None, None])
    plt.grid()
    plt.show()

    return


def extract_all_mvcs(
    verbose: bool,
    show_plots: bool,
) -> dict:

    t_start_s = time.time()

    # structure to store all computed MVCs
    mvc_init_value_v = 0.0  # 0.0V as initialization value
    mvc_v_dict = init_all_mvcs_dict(mvc_init_value_v)

    for idx_subj, idx_sess, idx_finger, force_direction in product(
        range(hy.NUM_SUBJECTS),
        range(hy.NUM_SESSIONS),
        range(hy.NUM_FINGERS),
        hy.ForceDirection,
    ):

        # load data from the MVC trial
        force_allfingers_v = hy.load_force(
            dataset=hy.Dataset.MVC,
            idx_subject=idx_subj,
            idx_session=idx_sess,
            idx_finger=idx_finger,
            idx_combination=None,
            force_direction=force_direction,
            idx_trial=None,
        )
        force_finger_v = force_allfingers_v[idx_finger]

        # extract mvc
        mvc_v = extract_mvc(force_finger_v, force_direction)
        mvc_v_dict['subject'][idx_subj]['session'][idx_sess]['direction'][
            force_direction.value][idx_finger] = mvc_v

        # plot for inspection
        # (the visualization function can be improved)
        if show_plots:
            show_mvc_plot(force_allfingers_v, force_direction, mvc_v)

        del mvc_v  # delete the old one for caution

    t_end_s = time.time()
    deltat_s = t_end_s - t_start_s
    if verbose:
        print(f"\nTime taken for extracting all MVCs: {deltat_s:.3f}s\n")

    return mvc_v_dict


# As per the toolbox, the rescale is done at the very beginning on each
# recording, prior to the regression training and validation.

# NB.
# SINCE THE SCALE DEPENDS ON THE SIGN, THIS RESCALING CAN ONLY BE APPLIED TO A
# GROUND-TRUTH FORCE. Estimated forces will be already in MVC units, because
# training is done after rescaling, thus in MVC units.

def rescale_force_volt2mvc(
    force_v: np.ndarray[np.float32],
    mvc_ext_v: np.ndarray[np.float32],
    mvc_flex_v: np.ndarray[np.float32],
) -> np.ndarray[np.float32]:

    num_ch, num_samples = force_v.shape
    assert num_ch == hy.NUM_FINGERS
    assert mvc_ext_v.shape == (hy.NUM_FINGERS,)
    assert mvc_flex_v.shape == (hy.NUM_FINGERS,)
    assert np.all(mvc_ext_v > 0.0)
    assert np.all(mvc_flex_v > 0.0)

    mask_ext = force_v > 0.0  # extension segments
    mask_flex = ~ mask_ext  # flexion segments

    # this safely avoids Python's aliasing
    force_mvc = np.zeros((num_ch, num_samples), dtype=np.float32)

    # loop instead of broadcast, to use boolean indexing
    for idf in range(hy.NUM_FINGERS):  # idf stands for index of finger

        # extension
        force_mvc[idf, mask_ext[idf]] = \
            force_v[idf, mask_ext[idf]] / mvc_ext_v[idf]

        # flexion
        force_mvc[idf, mask_flex[idf]] = \
            force_v[idf, mask_flex[idf]] / mvc_flex_v[idf]

    return force_mvc


def main() -> None:
    pass


if __name__ == '__main__':
    main()
