from __future__ import annotations
from itertools import product
import os
from pathlib import Path
import pickle

import numpy as np
import torch.optim

from ..hyser import hyser as hy
from ..hyser import mvc
from ..learning import settings as learningsettings
from ..learning.models import TinierNet8
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
        hy.concatenate_paired_segments(hdsemg_v_wholeday, force_v_wholeday)

    return hdsemg_v_wholeday, force_v_wholeday


def inference_on_onedof_ndof_random(idx_subject, model):

    wholeinference_dict = {
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

    print('\n\nValidating on ONEDOF')
    
    for idx_day in range(hy.NUM_SESSIONS):
        wholeinference_dict['onedof']['day'][idx_day] = {'finger': {}}
        for idx_finger in range(hy.NUM_FINGERS):
            wholeinference_dict['onedof']['day'][idx_day]['finger'][idx_finger] = {'trial': {}}
            for idx_trial in range(hy.NUM_TRIALS_ONEDOF):

                mvc_ext_v_day = mvc.MVC_V_DICT['subject'][idx_subject]['session'][idx_day]['direction'][hy.ForceDirection.EXTENSION.value]
                mvc_flex_v_day = mvc.MVC_V_DICT['subject'][idx_subject]['session'][idx_day]['direction'][hy.ForceDirection.FLEXION.value]

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
                y = mvc.rescale_force_volt2mvc(y, mvc_ext_v_day, mvc_flex_v_day)

                yout = learn.do_inference(x, model)
                wholeinference_dict['onedof']['day'][idx_day]['finger'][idx_finger]['trial'][idx_trial] = \
                    good.compute_regression_metrics(y[:, learn.WINDOW - 1 :: learn.SLIDE], yout.T)
                del x, y, yout

    # ----------------------------------------------------------------------- #

    # NDOF
                
    print('\n\nValidating on NDOF')

    for idx_day in range(hy.NUM_SESSIONS):
        wholeinference_dict['ndof']['day'][idx_day] = {'combination': {}}
        for idx_combination in range(hy.NUM_COMBINATIONS_NDOF):
            wholeinference_dict['ndof']['day'][idx_day]['combination'][idx_combination] = {'trial': {}}
            for idx_trial in range(hy.NUM_TRIALS_NDOF):

                mvc_ext_v_day = mvc.MVC_V_DICT['subject'][idx_subject]['session'][idx_day]['direction'][hy.ForceDirection.EXTENSION.value]
                mvc_flex_v_day = mvc.MVC_V_DICT['subject'][idx_subject]['session'][idx_day]['direction'][hy.ForceDirection.FLEXION.value]

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
                y = mvc.rescale_force_volt2mvc(y, mvc_ext_v_day, mvc_flex_v_day)
    
                yout = learn.do_inference(x, model)
                wholeinference_dict['ndof']['day'][idx_day]['combination'][idx_combination]['trial'][idx_trial] = \
                    good.compute_regression_metrics(y[:, learn.WINDOW - 1 :: learn.SLIDE], yout.T)
                del x, y, yout

    # ----------------------------------------------------------------------- #
    
    # RANDOM
    
    print('\n\nValidating on RANDOM')
    
    for idx_day in range(hy.NUM_SESSIONS):
        wholeinference_dict['random']['day'][idx_day] = {'trial': {}}
        for idx_trial in range(hy.NUM_TRIALS_RANDOM):

            mvc_ext_v_day = mvc.MVC_V_DICT['subject'][idx_subject]['session'][idx_day]['direction'][hy.ForceDirection.EXTENSION.value]
            mvc_flex_v_day = mvc.MVC_V_DICT['subject'][idx_subject]['session'][idx_day]['direction'][hy.ForceDirection.FLEXION.value]

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
            y = mvc.rescale_force_volt2mvc(y, mvc_ext_v_day, mvc_flex_v_day)

            yout = learn.do_inference(x, model)
            wholeinference_dict['random']['day'][idx_day]['trial'][idx_trial] = \
                good.compute_regression_metrics(y[:, learn.WINDOW - 1 :: learn.SLIDE], yout.T)
            del x, y, yout
    
    # ----------------------------------------------------------------------- #

    return wholeinference_dict


def experiment_one_subject(
    idx_subject: int,
    learning_mode: str,
) -> dict:
    
    assert idx_subject in range(hy.NUM_SUBJECTS)
    assert learning_mode in ['baseline', 'online']

    model = TinierNet8(num_ch_in=64, num_ch_out=hy.NUM_CHANNELS_FORCE)
    # model.half()

    if learning_mode == 'baseline':

        loadermode_train = learn.LoaderMode.TRAINING_RANDOMIZED
        minibatch_train = 32
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.0001, weight_decay=0.0)
        num_epochs = 8

    elif learning_mode == 'online':
    
        loadermode_train = learn.LoaderMode.TRAINING_SEQUENTIAL
        minibatch_train = 1
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.000176, weight_decay=0.0)
        num_epochs = 1

    else:
        raise NotImplementedError



    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    results_onesubj_dict = {
        'stage': {
            0: {},
            1: {},
            2: {},
        }
    }

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    # determine MVCs to rescale forces
    # TODO: fast but redundant, repeated!
    mvc.MVC_V_DICT = mvc.extract_all_mvcs(verbose=True, show_plots=False)
    # TODO: write a function to unpack
    mvc_ext_v_train = mvc.MVC_V_DICT['subject'][idx_subject]['session'][0]['direction'][hy.ForceDirection.EXTENSION.value]
    mvc_flex_v_train = mvc.MVC_V_DICT['subject'][idx_subject]['session'][0]['direction'][hy.ForceDirection.FLEXION.value]
    mvc_ext_v_valid = mvc.MVC_V_DICT['subject'][idx_subject]['session'][1]['direction'][hy.ForceDirection.EXTENSION.value]
    mvc_flex_v_valid = mvc.MVC_V_DICT['subject'][idx_subject]['session'][1]['direction'][hy.ForceDirection.FLEXION.value]
    
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    # STAGE 0
    
    # load whole one-dof and rescale forces to units of MVC
    xtrain, ytrain = load_concat_day_of_hyser_dataset(
        hy.Dataset.ONEDOF, idx_subject, idx_day=0)
    xvalid, yvalid = load_concat_day_of_hyser_dataset(
        hy.Dataset.ONEDOF, idx_subject, idx_day=1)
    ytrain = mvc.rescale_force_volt2mvc(
        ytrain, mvc_ext_v_train, mvc_flex_v_train)
    yvalid = mvc.rescale_force_volt2mvc(
        yvalid, mvc_ext_v_valid, mvc_flex_v_valid)

    # train on whole one-dof
    training_summary_dict = learn.do_training(
        xtrain, ytrain, xvalid, yvalid, model,
        loadermode_train=loadermode_train,
        optimizer=optimizer,
        minibatch_train=minibatch_train,
        num_epochs=num_epochs,
    )
    del xtrain, ytrain, xvalid, yvalid

    model = training_summary_dict['model']
    # test on everything
    results_onesubj_dict['stage'][0] = \
        inference_on_onedof_ndof_random(idx_subject, model)
    # ### del model

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    # STAGE 1

    # load whole n-dof and rescale forces to units of MVC
    xtrain, ytrain = load_concat_day_of_hyser_dataset(
        hy.Dataset.NDOF, idx_subject, idx_day=0)
    xvalid, yvalid = load_concat_day_of_hyser_dataset(
        hy.Dataset.NDOF, idx_subject, idx_day=1)
    ytrain = mvc.rescale_force_volt2mvc(
        ytrain, mvc_ext_v_train, mvc_flex_v_train)
    yvalid = mvc.rescale_force_volt2mvc(
        yvalid, mvc_ext_v_valid, mvc_flex_v_valid)

    # train on whole one-dof
    training_summary_dict = learn.do_training(
        xtrain, ytrain, xvalid, yvalid, model,
        loadermode_train=loadermode_train,
        optimizer=optimizer,
        minibatch_train=minibatch_train,
        num_epochs=num_epochs,
    )
    del xtrain, ytrain, xvalid, yvalid

    model = training_summary_dict['model']
    # test on everything
    results_onesubj_dict['stage'][1] = \
        inference_on_onedof_ndof_random(idx_subject, model)
    # ### del model

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    # STAGE 2

    # load whole random and rescale forces to units of MVC
    xtrain, ytrain = load_concat_day_of_hyser_dataset(
        hy.Dataset.RANDOM, idx_subject, idx_day=0)
    xvalid, yvalid = load_concat_day_of_hyser_dataset(
        hy.Dataset.RANDOM, idx_subject, idx_day=1)
    ytrain = mvc.rescale_force_volt2mvc(
        ytrain, mvc_ext_v_train, mvc_flex_v_train)
    yvalid = mvc.rescale_force_volt2mvc(
        yvalid, mvc_ext_v_valid, mvc_flex_v_valid)

    # train on whole random
    training_summary_dict = learn.do_training(
        xtrain, ytrain, xvalid, yvalid, model,
        loadermode_train=loadermode_train,
        optimizer=optimizer,
        minibatch_train=minibatch_train,
        num_epochs=num_epochs,
    )
    del xtrain, ytrain, xvalid, yvalid

    model = training_summary_dict['model']
    # test on everything
    results_onesubj_dict['stage'][2] = \
        inference_on_onedof_ndof_random(idx_subject, model)
    del model

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    return results_onesubj_dict


def save_results_dict(
    results_dict: dict,
    filename: str,
) -> None:

    dict_to_dump = {'results': results_dict}  # add an outer key

    DST_DIR = './results'  # destionation directory
    Path(DST_DIR).mkdir(parents=True, exist_ok=True)
    path = os.path.join(DST_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(dict_to_dump, f)

    return


def experiment_all_subjects(
    learning_mode: str,
    results_filename: str,
    ) -> dict:

    assert learning_mode in ['baseline', 'online']
    assert isinstance(results_filename, str)

    # ----------------------------------------------------------------------- #
    # set the seeds of all the random generators used
    learningsettings.set_reproducibility()
    # ----------------------------------------------------------------------- #

    # initialize the results dictionary, the structure to store the results
    results_allsubj_dict = {
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

        results_allsubj_dict['subject'][idx_subject] = \
            experiment_one_subject(
                idx_subject=idx_subject,
                learning_mode=learning_mode,
            )

        # save all results after each subject
        save_results_dict(
            results_dict=results_allsubj_dict,
            filename=results_filename,
        )

    return results_allsubj_dict


def main() -> None:
    pass


if __name__ == "__main__":
    main()
