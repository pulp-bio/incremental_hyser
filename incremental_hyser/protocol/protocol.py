from __future__ import annotations
from itertools import product
from pathlib import Path
import pickle

import numpy as np

from ..hyser import hyser as hy
from ..hyser import mvc

from ..learning import settings as learningsettings
from ..learning import learning as learn


def concatenate_day_of_hyser_dataset(
    dataset: hy.Dataset,
    idx_subject: int,
    idx_session: int,
) -> tuple[
    np.ndarray[np.float32],  # hdsemg_v_wholeday
    np.ndarray[np.float32],  # force_v_wholeday
]:
    
    # assert
    assert dataset in hy.Dataset
    assert idx_subject in range(hy.NUM_SUBJECTS)
    assert idx_session in range(hy.NUM_SESSIONS)

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
            dataset=dataset
            idx_subject=idx_subject,
            idx_session=idx_session,
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

    result_dict = {
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
    
    # onedof
    for idx_day in range(hy.NUM_SESSIONS):


    # ndof


    # random
    for idx_day, idx_trial in product(range(hy.NUM_SESSIONS), range(hy.NUM_TRIALS_RANDOM)):

        x, y = load... # MOVE THE FORCE NORMALIZATION INSIDE THE LOAD!
        '''
        hdsemg_v_i, force_v_i = hy.load_hdsemg_and_force(
            dataset=dataset
            idx_subject=idx_subject,
            idx_session=idx_session,
            pr_task_type=None,  # hardcoded
            hdsemg_signal_type=hy.SignalType.PREPROCESS,  # hardcoded
            idx_finger=idx_finger,
            idx_combination=idx_combination,
            force_direction=None,  # hardcoded
            idx_trial=idx_trial,
        )
        '''

        yout = do_inference()

        results_dict['random']['day'][idx_day] = compute_regression_metrics()





    







    return


def experiment_one_subject(

    idx_subject: int,

) -> dict:

    # ----------------------------------------------------------------------- #
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
    # ----------------------------------------------------------------------- #
    
    # adjust sampling
    # TODO: MAYBE SIMPLY SIMULATE WITH STEP AS FORCE SAMPLING? MUCH FASTER!
    # x_post_train = hy.downsample_myo_as_force(x_post_train)
    # x_post_valid = hy.downsample_myo_as_force(x_post_valid)

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    # load whole one-dof
    # train on whole one-dof
    # test on everything

    # load ndof, day 1
    # train
    # test on everything

    # load random, day 1
    # train
    # test on everything


    return results_dict_one_subj


def save_results_dict(
    results_dict: dict,
    results_dict_dst_filepath: str,
) -> dict:

    dict_to_dump = {'results': results_dict}  # add an outer key

    # Path(RESULTS_DIR_PATH).mkdir(parents=True, exist_ok=True)
    with open(results_dict_dst_filepath, 'wb') as f:
        pickle.dump(dict_to_dump, f)

    return


def experiment_all_subjects(

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
            experiment_one_subject(idx_subject=idx_subject)


        # save all results after each subject
        save_results_dict(results_dict, results_dict_dst_filepath)

    return results_dict


def main() -> None:
    pass


if __name__ == "__main__":
    main()
