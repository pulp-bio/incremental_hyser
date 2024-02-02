# %%
import itertools
from pathlib import Path
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ztbiocas.datasets import uniboinail as ui
from ztbiocas.learning import mlpuniboinail as mui
from ztbiocas.learning import settings as learnset
from ztbiocas.learning import learning as learn
from ztbiocas.analysis import goodness as good

# %%
NUM_REPETITIONS = 1

DOWNSAMPLING_FACTOR = 1

NUM_TRAIN_REPETITIONS = 5

NUM_EPOCHS_FP = 4

RESULTS_FILENAME = 'results_multitrain_replication.pkl'
RESULTS_DIR_PATH = './results/'
RESULTS_FILE_FULLPATH = RESULTS_DIR_PATH + RESULTS_FILENAME

# %%
# structure for storing the results

results = {
    'num_repetitions': NUM_REPETITIONS,
    'repetition': {},
}


for idx_repetition in range(NUM_REPETITIONS):
    results['repetition'][idx_repetition] = {'subject': {}}

    for idx_subject in range(ui.NUM_SUBJECTS):

        results['repetition'][idx_repetition]['subject'][idx_subject] = {'day': {}}

        for idx_day in range(ui.NUM_DAYS):

            results['repetition'][idx_repetition]['subject'][idx_subject]['day'][idx_day] = {'posture': {}}

            for idx_valid_posture in range(ui.NUM_POSTURES):

                results['repetition'][idx_repetition]['subject'][idx_subject]['day'][idx_day]['posture'][idx_valid_posture] = {
                    # just classification metrics, no models or labels
                    'training': {},  # classification metrics dictionary
                    'validation': {},  # classification metrics dictionary
                }

# %%
learnset.set_reproducibility()


for idx_repetition, idx_subject, idx_day in itertools.product(
    range(NUM_REPETITIONS), range(ui.NUM_SUBJECTS), range(ui.NUM_DAYS)
):
    
    # ----------------------------------------------------------------------- #

    # print a header
    print(
        f"\n"
        f"------------------------------------------------------------------\n"
        f"REPETITION\t{idx_repetition + 1 :d}/{NUM_REPETITIONS:d}\n"
        f"SUBJECT\t{idx_subject + 1 :d}/{ui.NUM_SUBJECTS:d}\n"
        f"DAY\t{idx_day + 1 :d}/{ui.NUM_DAYS:d}\n"
        f"(all indices are one-based)\n"
        f"------------------------------------------------------------------\n"
        f"\n"
    )

    # ----------------------------------------------------------------------- #

    # load training data

    xtrain_list = []
    ytrain_list = []

    for idx_train_posture in range(ui.NUM_POSTURES):
        
        train_session_data_dict = ui.load_session(
            idx_subject, idx_day, idx_train_posture)

        emg_train = train_session_data_dict['emg']
        relabel_train = train_session_data_dict['relabel']
        gesture_counter_train = train_session_data_dict['gesture_counter']
        del train_session_data_dict

        # "_p" stands for single posture
        xtrain_p, ytrain_p, _, _ = ui.split_into_calib_and_valid(
            emg=emg_train,
            relabel=relabel_train,
            gesture_counter=gesture_counter_train,
            num_calib_repetitions=NUM_TRAIN_REPETITIONS,
        )
        del emg_train, relabel_train, gesture_counter_train

        # add to the lists
        xtrain_list.append(xtrain_p)
        ytrain_list.append(ytrain_p)
        del xtrain_p, ytrain_p

    # concatenate into single arrays
    xtrain = np.concatenate(xtrain_list, axis=1)
    ytrain = np.concatenate(ytrain_list, axis=0)
    del xtrain_list, ytrain_list

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

    mlp = mui.MLPUniboINAIL(
        num_input=ui.NUM_CHANNELS, num_hidden=8, num_output=ui.NUM_CLASSES)
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
    del xtrain_pc, ytrain
    
    # ----------------------------------------------------------------------- #

    for idx_valid_posture in range(ui.NUM_POSTURES):

        # ------------------------------------------------------------------- #

        # print a header
        print(
            f"\n"
            f"--------------------------------------------------------------\n"
            f"VALIDATION ON POSTURE {idx_valid_posture + 1 :d}\n"
            f"--------------------------------------------------------------\n"
            f"\n"
        )

        # ------------------------------------------------------------------- #
        
        # load validation data

        valid_session_data_dict = ui.load_session(
            idx_subject, idx_day, idx_valid_posture)

        emg_valid = valid_session_data_dict['emg']
        relabel_valid = valid_session_data_dict['relabel']
        gesture_counter_valid = valid_session_data_dict['gesture_counter']
        del valid_session_data_dict

        # "_p" stands for single posture
        xtrain_p, ytrain_p, xvalid, yvalid = ui.split_into_calib_and_valid(
            emg=emg_valid,
            relabel=relabel_valid,
            gesture_counter=gesture_counter_valid,
            num_calib_repetitions=NUM_TRAIN_REPETITIONS,
        )
        del emg_valid, relabel_valid, gesture_counter_valid

        # ------------------------------------------------------------------- #

        # preprocessing

        xtrain_p = xtrain_p[:, ::DOWNSAMPLING_FACTOR]
        ytrain_p = ytrain_p[::DOWNSAMPLING_FACTOR]
        xvalid = xvalid[:, ::DOWNSAMPLING_FACTOR]
        yvalid = yvalid[::DOWNSAMPLING_FACTOR]

        xtrain_p_standardscaled = stdscaler_train.transform(xtrain_p.T).T
        xvalid_standardscaled = stdscaler_train.transform(xvalid.T).T
        del xtrain_p, xvalid
        xtrain_p_pc = pca_train.transform(xtrain_p_standardscaled.T).T
        xvalid_pc = pca_train.transform(xvalid_standardscaled.T).T
        del xtrain_p_standardscaled, xvalid_standardscaled

        # ------------------------------------------------------------------- #

        # MLP inference
        
        yout_train_p = learn.do_inference(xtrain_p_pc, mlp)
        yout_valid = learn.do_inference(xvalid_pc, mlp)
        del xtrain_p_pc, xvalid_pc

        metrics_train_p = good.compute_classification_metrics(ytrain_p, yout_train_p)
        metrics_valid = good.compute_classification_metrics(yvalid, yout_valid)

        print("\n\n")
        print("On training repetitions:")
        print(metrics_train_p)
        print("On validation repetitions:")
        print(metrics_valid)
        print("\n\n")
        
        # ------------------------------------------------------------------- #

        # store results
        results['repetition'][idx_repetition]['subject'][idx_subject]['day'][idx_day]['posture'][idx_valid_posture]['training'] = metrics_train_p
        results['repetition'][idx_repetition]['subject'][idx_subject]['day'][idx_day]['posture'][idx_valid_posture]['validation'] = metrics_valid
        
        # ------------------------------------------------------------------- #

        # save to file
        # save the updated results dictionary after each validation
        results_outer_dict = {'results': results}
        Path(RESULTS_DIR_PATH).mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE_FULLPATH, 'wb') as f:
            pickle.dump(results_outer_dict, f)

        # ------------------------------------------------------------------- #

# %%
"""
1/4		0.3015	0.8840		none	none		5.4
2/4		0.2468	0.9058		none	none		4.7
3/4		0.2405	0.9093		none	none		4.7
4/4		0.2357	0.9116		none	none		4.7
"""

"""
1/4		0.3015	0.8840		none	none		5.3
2/4		0.2468	0.9058		none	none		4.9
3/4		0.2405	0.9093		none	none		5.7
4/4		0.2357	0.9116		none	none		4.8
"""


