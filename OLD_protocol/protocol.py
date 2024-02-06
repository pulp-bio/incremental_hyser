from __future__ import annotations
import pickle
import random

import numpy as np

from ..hyser import hyser as hy
from ..hyser import mvc
# from ..spikification import spikification as spk
# from ..regression import regression as rgr


def set_random_seeds() -> None:

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)

    return


def concat_trials_of_sess_of_random(  # hardcoded for the HYSER random dataset
    idx_subject: int,
    idx_session: int,
) -> tuple[
    np.ndarray[np.float32],  # hdsemg_v_alltrials
    np.ndarray[np.float32],  # force_v_alltrials
]:

    assert idx_subject in range(hy.NUM_SUBJECTS)
    assert idx_session in range(hy.NUM_SESSIONS)

    hdsemg_trials_list = []
    force_trials_list = []

    for idx_trial in range(hy.NUM_TRIALS_RANDOM):

        hdsemg_v_trial, force_v_trial = hy.load_hdsemg_and_force(
            dataset=hy.Dataset.RANDOM,  # hardcoded
            idx_subject=idx_subject,
            idx_session=idx_session,
            pr_task_type=None,
            hdsemg_signal_type=hy.SignalType.PREPROCESS,
            idx_finger=None,
            idx_combination=None,
            force_direction=None,
            idx_trial=idx_trial,
        )
        hdsemg_trials_list.append(hdsemg_v_trial)
        del hdsemg_v_trial
        force_trials_list.append(force_v_trial)
        del force_v_trial

    # concatenate
    hdsemg_v_alltrials, force_v_alltrials = \
        hy.concatenate_trials(hdsemg_trials_list, force_trials_list)
    del hdsemg_trials_list, force_trials_list

    return hdsemg_v_alltrials, force_v_alltrials


IDX_SESSION_TRAINING = 0  # train on session 1
IDX_SESSION_VALIDATION = 1  # validate on session 2

assert IDX_SESSION_TRAINING in range(hy.NUM_SESSIONS)
assert IDX_SESSION_VALIDATION in range(hy.NUM_SESSIONS)


def compose_trainining_and_validation_sets(
    idx_subject: int,
) -> tuple[
    np.ndarray[np.float32],  # hdsemg_v_train
    np.ndarray[np.float32],  # force_v_train
    np.ndarray[np.float32],  # hdsemg_v_valid
    np.ndarray[np.float32],  # force_v_valid
]:

    hdsemg_v_train, force_v_train = \
        concat_trials_of_sess_of_random(idx_subject, IDX_SESSION_TRAINING)

    hdsemg_v_valid, force_v_valid = \
        concat_trials_of_sess_of_random(idx_subject, IDX_SESSION_VALIDATION)

    return hdsemg_v_train, force_v_train, hdsemg_v_valid, force_v_valid


def protocol_on_single_subject(

    idx_subject: int,

    gain_per_volt: float,
    x_init: float,
    x_reset: float,
    taupre_s: float,
    trefr_s: float,
    taupost_s: float,
    dt_sim_s: float,
    dt_monitors_pre_s: float,
    dt_monitor_post_s: float,
    report_stdstream: str | None,
    report_period_s: float | None,

    downsamp: np.uint32,
    alpha: float,

) -> dict:

    hdsemg_v_train, force_v_train, hdsemg_v_valid, force_v_valid = \
        compose_trainining_and_validation_sets(idx_subject)

    # ----------------------------------------------------------------------- #

    # determine MVCs and rescale forces
    # TODO: fast but redundant, repeated!
    mvc_v_dict = mvc.extract_all_mvcs(verbose=True, show_plots=False)
    # TODO: write a function to unpack

    # TODO: BETTER TO RESCALE PAIRWISE AFER INFERENCE!
    # OR MAYBE TRY BOTH WAYS AND SEE IF ACCURACY CHANGES

    mvc_ext_v_train = mvc_v_dict['subject'][idx_subject]['session'][
        IDX_SESSION_TRAINING]['direction'][hy.ForceDirection.EXTENSION.value]
    mvc_flex_v_train = mvc_v_dict['subject'][idx_subject]['session'][
        IDX_SESSION_TRAINING]['direction'][hy.ForceDirection.FLEXION.value]
    mvc_ext_v_valid = mvc_v_dict['subject'][idx_subject]['session'][
        IDX_SESSION_VALIDATION]['direction'][hy.ForceDirection.EXTENSION.value]
    mvc_flex_v_valid = mvc_v_dict['subject'][idx_subject]['session'][
        IDX_SESSION_VALIDATION]['direction'][hy.ForceDirection.FLEXION.value]

    force_mvc_train = mvc.rescale_force_volt2mvc(
        force_v_train, mvc_ext_v_train, mvc_flex_v_train)
    del force_v_train
    force_mvc_valid = mvc.rescale_force_volt2mvc(
        force_v_valid, mvc_ext_v_train, mvc_flex_v_train)
    del force_v_valid

    # ----------------------------------------------------------------------- #

    # spikification

    # spikifiy the training data

    _, _, _, x_post_train = spk.spikify(
        hdsemg_v=hdsemg_v_train,
        fs_hz=hy.FS_HDSEMG,
        # lowcut_hz=spk.FREQ_CUT_LO_HZ,
        # highcut_hz=spk.FREQ_CUT_HI_HZ,
        # order=spk.ORDER,
        gain_per_volt=gain_per_volt,
        x_init=x_init,
        x_reset=x_reset,
        taupre_s=taupre_s,
        trefr_s=trefr_s,
        taupost_s=taupost_s,
        dt_sim_s=dt_sim_s,
        dt_monitors_pre_s=dt_monitors_pre_s,
        dt_monitor_post_s=dt_monitor_post_s,
        report_stdstream=report_stdstream,
        report_period_s=report_period_s,
    )
    del hdsemg_v_train

    # spikifiy the validation data

    _, _, _, x_post_valid = spk.spikify(
        hdsemg_v=hdsemg_v_valid,
        fs_hz=hy.FS_HDSEMG,
        # lowcut_hz=spk.FREQ_CUT_LO_HZ,
        # highcut_hz=spk.FREQ_CUT_HI_HZ,
        # order=spk.ORDER,
        gain_per_volt=gain_per_volt,
        x_init=x_init,
        x_reset=x_reset,
        taupre_s=taupre_s,
        trefr_s=trefr_s,
        taupost_s=taupost_s,
        dt_sim_s=dt_sim_s,
        dt_monitors_pre_s=dt_monitors_pre_s,
        dt_monitor_post_s=dt_monitor_post_s,
        report_stdstream=report_stdstream,
        report_period_s=report_period_s,
    )
    del hdsemg_v_valid

    # adjust sampling
    # TODO: MAYBE SIMPLY SIMULATE WITH STEP AS FORCE SAMPLING? MUCH FASTER!
    # x_post_train = hy.downsample_myo_as_force(x_post_train)
    # x_post_valid = hy.downsample_myo_as_force(x_post_valid)

    # fit regression
    regressor = rgr.train_multiregressor(
        x_post_train, force_mvc_train, downsamp, alpha)
    metrics_train = rgr.evaluate_multiregressor(
        regressor, x_post_train, force_mvc_train, downsamp)
    metrics_valid = rgr.evaluate_multiregressor(
        regressor, x_post_valid, force_mvc_valid, downsamp)

    # store the results orderly into a dictionary
    results_dict_single_subj = {
        'regressor': regressor,
        'training': metrics_train,
        'validation': metrics_valid,
    }

    return results_dict_single_subj


def save_results_dict(
    results_dict: dict,
    results_dict_dst_filepath: str,
) -> dict:

    dict_to_dump = {'results': results_dict}  # add an outer key
    with open(results_dict_dst_filepath, 'wb') as f:
        pickle.dump(dict_to_dump, f)

    return


def regression_experiment_on_every_subject(
    results_dict_dst_filepath: str,

    gain_per_volt: float,
    x_init: float,
    x_reset: float,
    taupre_s: float,
    trefr_s: float,
    taupost_s: float,
    dt_sim_s: float,
    dt_monitors_pre_s: float,
    dt_monitor_post_s: float,
    report_stdstream: str | None,
    report_period_s: float | None,

    downsamp: np.uint32,
    alpha: float,

) -> dict:

    # ----------------------------------------------------------------------- #
    # set the seeds of all the random generators used
    set_random_seeds()
    # ----------------------------------------------------------------------- #

    # initialize the results dictionary, the structure to store the results
    results_dict = {'subject': {}}
    # for idx_subject in range(hy.NUM_SUBJECTS):
    #    results_dict['subject'][idx_subject] = {
    #        'regressor': None,
    #        'training': {},
    #        'validation': {},
    #    }

    for idx_subject in range(hy.NUM_SUBJECTS):

        # ------------------------------------------------------------------- #
        print(
            f"\n------------------------------------------------------------\n"
            f"SUBJECT {idx_subject + 1}/{hy.NUM_SUBJECTS}"
            f"\n------------------------------------------------------------\n"
        )
        # ------------------------------------------------------------------- #

        results_dict['subject'][idx_subject] = protocol_on_single_subject(

            idx_subject=idx_subject,

            gain_per_volt=gain_per_volt,
            x_init=x_init,
            x_reset=x_reset,
            taupre_s=taupre_s,
            trefr_s=trefr_s,
            taupost_s=taupost_s,
            dt_sim_s=dt_sim_s,
            dt_monitors_pre_s=dt_monitors_pre_s,
            dt_monitor_post_s=dt_monitor_post_s,
            report_stdstream=report_stdstream,
            report_period_s=report_period_s,

            downsamp=downsamp,
            alpha=alpha,
        )

        # save all results after each subject
        save_results_dict(results_dict, results_dict_dst_filepath)

    return results_dict


def main() -> None:
    pass


if __name__ == '__main__':
    main()
