from __future__ import annotations
from itertools import product
import os
from pathlib import Path
import pickle

import numpy as np

from ..hyser import hyser as hy
from ..hyser import mvc

from ..learning import settings as learningsettings
from ..learning import learning as learn
from ..learning import goodness as good


def load_concat_day_of_hyser_dataset(
    dataset: hy.Dataset,
    idx_subject: int,
    idx_day: int,
) -> tuple[
    np.ndarray[np.float32],  # hdsemg_v_wholeday
    np.ndarray[np.float32],  # force_v_wholeday
]:
    
    # assert
    assert dataset in hy.Dataset
    assert idx_subject in range(hy.NUM_SUBJECTS)
    assert idx_day in range(hy.NUM_SESSIONS)

    if dataset is hy.Dataset.ONEDOF :
        cartprod_of_dataset = product(
            range(hy.NUM_FINGERS),  # finger
            [None],  # combination
            range(hy.NUM_TRIALS_ONEDOF)  # trial
        )
    elif dataset is hy.Dataset.NDOF :
        cartprod_of_dataset = product(
            [None],  # finger
            range(hy.NUM_COMBINATIONS_NDOF),  # combination
            range(hy.NUM_TRIALS_NDOF)  # trial
        )
    elif dataset is hy.Dataset.RANDOM:
        cartprod_of_dataset = product(
            [None],  # finger
            [None],  # combination
            range(hy.NUM_TRIALS_RANDOM)  # trial
        )
    else:
        raise ValueError

    hdsemg_v_wholeday = []
    force_v_wholeday = []

    for multi_idx in cartprod_of_dataset:

        idx_finger, idx_combination, idx_trial = multi_idx

        hdsemg_v_i, force_v_i = hy.load_hdsemg_and_force(
            dataset=dataset,
            idx_subject=idx_subject,
            idx_session=idx_day,
            pr_task_type=None,  # hardcoded
            hdsemg_signal_type=hy.SignalType.PREPROCESS,  # hardcoded
            idx_finger=idx_finger,
            idx_combination=idx_combination,
            force_direction=None,  # hardcoded
            idx_trial=idx_trial,
        )
        
        hdsemg_v_wholeday.append(hdsemg_v_i)
        del hdsemg_v_i
        force_v_wholeday.append(force_v_i)
        del force_v_i

    # concatenate
    hdsemg_v_wholeday, force_v_wholeday = \
        hy.concatenate_trials(hdsemg_v_wholeday, force_v_wholeday)

    return hdsemg_v_wholeday, force_v_wholeday


def inference_on_onedof_ndof_random(idx_subject, model):

    results_dict = {
        'onedof': {
            'day': {},
        },
        'ndof': {
            'day': {},
        },
        'random': {
            'day': {},
        },
    }
    
    # ----------------------------------------------------------------------- #
    
    # ONEDOF
    
    for idx_day in range(hy.NUM_SESSIONS):
        results_dict['onedof']['day'][idx_day] = {'finger': {}}
        for idx_finger in range(hy.NUM_FINGERS):
            results_dict['onedof']['day'][idx_day]['finger'][idx_finger] = {'trial': {}}
            for idx_trial in range(hy.NUM_TRIALS_ONEDOF):

                x, y = hy.load_hdsemg_and_force(
                    dataset=hy.Dataset.ONEDOF,
                    idx_subject=idx_subject,
                    idx_session=idx_day,
                    pr_task_type=None,  # hardcoded
                    hdsemg_signal_type=hy.SignalType.PREPROCESS,  # hardcoded
                    idx_finger=idx_finger,
                    idx_combination=None,
                    force_direction=None,  # hardcoded
                    idx_trial=idx_trial,
                )
                yout = learn.do_inference(x, model)
                results_dict['onedof']['day'][idx_day]['finger'][idx_finger]['trial'][idx_trial] = good.compute_regression_metrics(y, yout)
                del x, y, yout

    # ----------------------------------------------------------------------- #

    # NDOF

    for idx_day in range(hy.NUM_SESSIONS):
        results_dict['ndof']['day'][idx_day] = {'combination': {}}
        for idx_combination in range(hy.NUM_COMBINATIONS_NDOF):
            results_dict['onedof']['day'][idx_day]['combination'][idx_combination] = {'trial': {}}
            for idx_trial in range(hy.NUM_TRIALS_NDOF):

                x, y = hy.load_hdsemg_and_force(
                    dataset=hy.Dataset.NDOF,
                    idx_subject=idx_subject,
                    idx_session=idx_day,
                    pr_task_type=None,  # hardcoded
                    hdsemg_signal_type=hy.SignalType.PREPROCESS,  # hardcoded
                    idx_finger=None,
                    idx_combination=idx_combination,
                    force_direction=None,  # hardcoded
                    idx_trial=idx_trial,
                )
                yout = learn.do_inference(x, model)
                results_dict['ndof']['day'][idx_day]['combination'][idx_combination]['trial'][idx_trial] = good.compute_regression_metrics(y, yout)
                del x, y, yout

    # ----------------------------------------------------------------------- #
    
    # RANDOM
    
    for idx_day in range(hy.NUM_SESSIONS):
        results_dict['random']['day'][idx_day] = {'trial': {}}
        for idx_trial in range(hy.NUM_TRIALS_RANDOM):

            # MOVE THE FORCE NORMALIZATION INSIDE THE LOAD!

            x, y = hy.load_hdsemg_and_force(
                dataset=hy.Dataset.RANDOM,
                idx_subject=idx_subject,
                idx_session=idx_day,
                pr_task_type=None,  # hardcoded
                hdsemg_signal_type=hy.SignalType.PREPROCESS,  # hardcoded
                idx_finger=None,
                idx_combination=None,
                force_direction=None,  # hardcoded
                idx_trial=idx_trial,
            )
            yout = learn.do_inference(x, model)
            results_dict['random']['day'][idx_day]['trial'][idx_trial] = \
                good.compute_regression_metrics(y, yout)
            del x, y, yout
    
    # ----------------------------------------------------------------------- #

    return results_dict


def experiment_one_subject(
    idx_subject: int,
    minibatch_size: int,
    optimizer_str: str,
) -> dict:

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    results_dict_one_subj = {
        'stage': {
            0: {},
            1: {},
            2: {},
        }
    }

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    """
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
    # ----------------------------------------------------------------------- #
    
    # adjust sampling
    # TODO: MAYBE SIMPLY SIMULATE WITH STEP AS FORCE SAMPLING? MUCH FASTER!
    # x_post_train = hy.downsample_myo_as_force(x_post_train)
    # x_post_valid = hy.downsample_myo_as_force(x_post_valid)

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    # load whole one-dof
    xtrain, ytrain = load_concat_day_of_hyser_dataset(hy.Dataset.ONEDOF, idx_subject, idx_day=0)
    xvalid, yvalid = load_concat_day_of_hyser_dataset(hy.Dataset.ONEDOF, idx_subject, idx_day=0)
    # train on whole one-dof
    training_outcome = learn.do_training(xtrain, ytrain, xvalid, yvalid, ...)
    model = training_outcome['model']
    # test on everything
    inference_results_dict = inference_on_onedof_ndof_random(idx_subject, model)

    # load ndof, day 1
    # train
    # test on everything

    # load random, day 1
    # train
    # test on everything
    """

    return results_dict_one_subj


def save_results_dict(
    results_dict: dict,
    dir_name: str,
    filename: str,
) -> None:

    dict_to_dump = {'results': results_dict}  # add an outer key

    Path(dir_name).mkdir(parents=True, exist_ok=True)
    path = os.path.join(dir_name, filename)
    with open(path, 'wb') as f:
        pickle.dump(dict_to_dump, f)

    return


def experiment_all_subjects(
    input_channels: int,
    minibatch_size: int,
    optimizer_str: str,
    results_directory: str,
    results_filename: str,
) -> dict:

    # ----------------------------------------------------------------------- #
    # set the seeds of all the random generators used
    learningsettings.set_reproducibility()
    # ----------------------------------------------------------------------- #

    # initialize the results dictionary, the structure to store the results
    results_dict = {
        'subject': {},
    }
    
    for idx_subject in range(hy.NUM_SUBJECTS):

        # ------------------------------------------------------------------- #
        print(
            f"\n------------------------------------------------------------\n"
            f"SUBJECT {idx_subject + 1}/{hy.NUM_SUBJECTS}"
            f"\n------------------------------------------------------------\n"
        )
        # ------------------------------------------------------------------- #

        results_dict['subject'][idx_subject] = \
            experiment_one_subject(
                idx_subject=idx_subject,
                input_channels=input_channels,
                minibatch_size=minibatch_size,
                optimizer_str=optimizer_str,
            )

        # save all results after each subject
        save_results_dict(
            results_dict=results_dict,
            results_directory=results_directory,
            results_filename=results_filename,
        )

    return results_dict


def main() -> None:
    pass


if __name__ == "__main__":
    main()
