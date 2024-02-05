from itertools import product

import matplotlib.pyplot as plt

from incremental_hyser.hyser import hyser as hy

xraw, _ = hy.load_hdsemg_and_force(
    dataset=hy.Dataset.RANDOM,
    idx_subject=0,
    idx_session=0,
    pr_task_type=None,
    hdsemg_signal_type=hy.SignalType.RAW,
    idx_finger=None,
    idx_combination=None,
    force_direction=None,
    idx_trial=0,
)

xpre, _ = hy.load_hdsemg_and_force(
    dataset=hy.Dataset.RANDOM,
    idx_subject=0,
    idx_session=0,
    pr_task_type=None,
    hdsemg_signal_type=hy.SignalType.PREPROCESS,
    idx_finger=None,
    idx_combination=None,
    force_direction=None,
    idx_trial=0,
)

import pickle

data_dict = {
    'xraw': xraw,
    'xpre': xpre,
}

with open('data_dict.pkl', 'wb') as handle:
    pickle.dump(data_dict, handle)
