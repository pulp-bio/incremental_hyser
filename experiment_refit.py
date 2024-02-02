# %%
import itertools
import pickle
from pathlib import Path

import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

import ztbiocas.datasets.uniboinail as ui
import ztbiocas.learning.mlpuniboinail as mui
import ztbiocas.learning.learning as learn
from ztbiocas import protocol

# %%
NUM_REPETITIONS = 1

DOWNSAMPLING_FACTOR = 1

NUM_CALIB_REPETITIONS = 5

NUM_EPOCHS_FP = 4

RESULTS_FILENAME = 'results_refit_replication.pkl'
RESULTS_DIR_PATH = './results/'
RESULTS_FILE_FULLPATH = RESULTS_DIR_PATH + RESULTS_FILENAME

# %%
# structure for storing the results

results = {
    'num_repetitions': NUM_REPETITIONS,
    'repetition': {},
}

for idx_rep in range(NUM_REPETITIONS):
    results['repetition'][idx_rep] = {'subject': {}}

    for idx_subject in range(ui.NUM_SUBJECTS):

        results['repetition'][idx_rep]['subject'][idx_subject] = {'day': {}}

        for idx_day in range(ui.NUM_DAYS):

            results['repetition'][idx_rep]['subject'][idx_subject]['day'][idx_day] = {'reference_posture': {}}

            for idx_ref_posture in range(ui.NUM_POSTURES):

                results['repetition'][idx_rep]['subject'][idx_subject]['day'][idx_day]['reference_posture'][idx_ref_posture] = {'target_posture': {}}

                for idx_tgt_posture in range(ui.NUM_POSTURES):

                    results['repetition'][idx_rep]['subject'][idx_subject]['day'][idx_day]['reference_posture'][idx_ref_posture]['target_posture'][idx_tgt_posture] = {
                        'calibration': {
                            'frozen': {},
                            'refit': {},
                        },
                        'validation': {
                            'frozen': {},
                            'refit': {},
                        },
                    }

# %%
for idx_repetition, idx_subject, idx_day, idx_ref_posture in itertools.product(
    range(NUM_REPETITIONS),
    range(ui.NUM_SUBJECTS),
    range(ui.NUM_DAYS),
    range(ui.NUM_POSTURES)
):
    
    # ----------------------------------------------------------------------- #

    # print a header
    print(
        f"\n"
        f"------------------------------------------------------------------\n"
        f"REPETITION\t{idx_repetition + 1 :d}/{NUM_REPETITIONS:d}\n"
        f"SUBJECT\t{idx_subject + 1 :d}/{ui.NUM_SUBJECTS:d}\n"
        f"DAY\t{idx_day + 1 :d}/{ui.NUM_DAYS:d}\n"
        f"POSTURE\t{idx_ref_posture + 1 :d}/{ui.NUM_POSTURES:d} AS REFERENCE\n"
        f"(all indices are one-based)\n"
        f"------------------------------------------------------------------\n"
        f"\n"
    )

    # ----------------------------------------------------------------------- #

    # load training data
    train_session_data_dict = ui.load_session(
        idx_subject, idx_day, idx_ref_posture)
    xtrain = train_session_data_dict['emg']
    ytrain = train_session_data_dict['relabel']
    del train_session_data_dict

    # ----------------------------------------------------------------------- #

    # downsampling
    xtrain = xtrain[:, ::DOWNSAMPLING_FACTOR]
    ytrain = ytrain[::DOWNSAMPLING_FACTOR]

    # standard scaling and de-correlation, as preprocessing before training
    stdscaler_train = StandardScaler()
    xtrain_stdscaled = stdscaler_train.fit_transform(xtrain.T).T
    del xtrain
    pca_train = PCA(n_components=ui.NUM_CHANNELS, whiten=False)
    xtrain_pc = pca_train.fit_transform(xtrain_stdscaled.T).T
    del xtrain_stdscaled

    # ----------------------------------------------------------------------- #

    # MLP training and validation

    mlp = mui.MLPUniboINAIL(num_input=ui.NUM_CHANNELS, num_hidden=8, num_output=ui.NUM_CLASSES)
    mui.summarize(mlp, num_input=ui.NUM_CHANNELS)

    # full-precision training
    mlp, history, yout_train, yout_valid = learn.do_training(
        xtrain=xtrain_pc,
        ytrain=ytrain,
        model=mlp,
        xvalid=None,
        yvalid=None,
        num_epochs=NUM_EPOCHS_FP,
        mltask=learn.MLTask.CLASSIFICATION,
        num_classes=ui.NUM_CLASSES,
    )

    # ----------------------------------------------------------------------- #

    # "tgt posture" stands for "target posture"
    for idx_tgt_posture in range(ui.NUM_POSTURES):

        # ------------------------------------------------------------------- #

        # print a header
        print(
            f"\n"
            f"--------------------------------------------------------------\n"
            f"TARGET POSTURE {idx_tgt_posture + 1 :d}\n"
            f"(trained on {idx_ref_posture + 1 :d})\n"
            f"--------------------------------------------------------------\n"
            f"\n"
        )

        # ------------------------------------------------------------------- #
        # ------------------------------------------------------------------- #

        # Do the two experiments:
        # - no adaptation
        # - refit the PCA online

        # ------------------------------------------------------------------- #
        # ------------------------------------------------------------------- #

        # load calibration and validation data

        calibvalid_session_data_dict = ui.load_session(
            idx_subject, idx_day, idx_tgt_posture)

        emg_calibvalid = calibvalid_session_data_dict['emg']
        relabel_calibvalid = calibvalid_session_data_dict['relabel']
        gesture_counter_calibvalid = \
            calibvalid_session_data_dict['gesture_counter']
        del calibvalid_session_data_dict

        xcalib, ycalib, xvalid, yvalid = ui.split_into_calib_and_valid(
            emg=emg_calibvalid,
            relabel=relabel_calibvalid,
            gesture_counter=gesture_counter_calibvalid,
            num_calib_repetitions=NUM_CALIB_REPETITIONS,
        )
        del emg_calibvalid, relabel_calibvalid, gesture_counter_calibvalid

        # ------------------------------------------------------------------- #

        # downsampling
        # NB: frozen standard scaling is included in the function
        # calibration_experiment
        xcalib = xcalib[:, ::DOWNSAMPLING_FACTOR]
        xvalid = xvalid[:, ::DOWNSAMPLING_FACTOR]
        ycalib = ycalib[::DOWNSAMPLING_FACTOR]
        yvalid = yvalid[::DOWNSAMPLING_FACTOR]
        
        # ------------------------------------------------------------------- #

        t0 = time.time()

        metrics = protocol.calibration_experiment(
            xcalib=xcalib,
            ycalib=ycalib,
            xvalid=xvalid,
            yvalid=yvalid,
            beta=0.01,
            stdscaler_train=stdscaler_train,
            pca_train=pca_train,
            model=mlp,
        )

        t1 = time.time()
        print(t1 - t0)
        
        print('VALIDATION METRICS')
        print('unbalanced')
        print(metrics['validation']['frozen']['accuracy'])
        print(metrics['validation']['refit']['accuracy'])
        print('balanced')
        print(metrics['validation']['frozen']['balanced_accuracy'])
        print(metrics['validation']['refit']['balanced_accuracy'])

        # store results
        results['repetition'][idx_repetition]['subject'][idx_subject]['day'][idx_day]['reference_posture'][idx_ref_posture][
            'target_posture'][idx_tgt_posture]['calibration']['frozen'] = metrics['calibration']['frozen']
        results['repetition'][idx_repetition]['subject'][idx_subject]['day'][idx_day]['reference_posture'][idx_ref_posture][
            'target_posture'][idx_tgt_posture]['validation']['frozen'] = metrics['validation']['frozen']
        
        results['repetition'][idx_repetition]['subject'][idx_subject]['day'][idx_day]['reference_posture'][idx_ref_posture][
            'target_posture'][idx_tgt_posture]['calibration']['refit'] = metrics['calibration']['refit']
        results['repetition'][idx_repetition]['subject'][idx_subject]['day'][idx_day]['reference_posture'][idx_ref_posture][
            'target_posture'][idx_tgt_posture]['validation']['refit'] = metrics['validation']['refit']
        
        del metrics

        # ------------------------------------------------------------------- #
        
        # save to file
        # save the updated results dictionary after each validation
        results_outer_dict = {'results': results}
        Path(RESULTS_DIR_PATH).mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE_FULLPATH, 'wb') as f:
            pickle.dump(results_outer_dict, f)

# %%



