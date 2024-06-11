from __future__ import annotations
import enum

import wfdb
import numpy as np


NUM_SUBJECTS = 20
NUM_SESSIONS = 2
NUM_FINGERS = 5

NUM_CHANNELS_HDSEMG = 256
NUM_CHANNELS_FORCE = NUM_FINGERS
FS_HDSEMG = 2048
FS_FORCE = 100

# Numbers of PR dataset
NUM_PR_GESTURES = 34
NUM_TRIALS_PR_DYNAMIC = 204  # TODO: unclear: paper says 204, but are 202
NUM_TRIALS_PR_MAINTENANCE = 68

# Numbers of the MVC dataset
# (none in addition to the common ones)

# Numbers of the 1-DoF dataset
NUM_TRIALS_ONEDOF = 3

# Numbers of the n-DoF dataset
NUM_COMBINATIONS_NDOF = 15
NUM_TRIALS_NDOF = 2

# Numbers of the Random dataset
NUM_TRIALS_RANDOM = 5

# Duration of the recordings, in seconds
TIME_SEGM_PR_DYNAMIC_S = 1.0  # TODO: to be confirmed
TIME_SEGM_PR_MAINTENANCE_S = 3.0  # TODO: to be confirmed
TIME_SEGM_MVC_S = 10.0
TIME_SEGM_ONEDOF_S = 25.0
TIME_SEGM_NDOF_S = 25.0
TIME_SEGM_RANDOM_S = 25.0

# Duration of the recordings, in samples
# TODO: to be completed
NUM_SAMPLES_HDSEMG_MVC = int(FS_HDSEMG * TIME_SEGM_MVC_S)
NUM_SAMPLES_HDSEMG_RANDOM = int(FS_HDSEMG * TIME_SEGM_RANDOM_S)
NUM_SAMPLES_FORCE_MVC = int(FS_FORCE * TIME_SEGM_MVC_S)
NUM_SAMPLES_FORCE_RANDOM = int(FS_FORCE * TIME_SEGM_RANDOM_S)

# Local path of the dataset
HYSER_PATH_DOWNLOADED = '/scratch/zanghieri/hyser/data/downloaded/'
DIRNAME_PR = 'pr_dataset/'
DIRNAME_MVC = 'mvc_dataset/'
DIRNAME_ONEDOF = '1dof_dataset/'
DIRNAME_NDOF = 'ndof_dataset/'
DIRNAME_RANDOM = 'random_dataset/'


@enum.unique
class Dataset(enum.Enum):
    PR = 'pr'
    MVC = 'mvc'
    ONEDOF = '1dof'
    NDOF = 'ndof'
    RANDOM = 'random'


@enum.unique
class PRTaskType(enum.Enum):
    DYNAMIC = 'dynamic'
    MAINTENANCE = 'maintenance'


@enum.unique
class ForceDirection(enum.Enum):
    EXTENSION = 'extension'
    FLEXION = 'flexion'


@enum.unique
class SignalType(enum.Enum):
    RAW = 'raw'
    PREPROCESS = 'preprocess'
    FORCE = 'force'


def filepath_noext_from_indices(
    dataset: Dataset,
    idx_subject: int,
    idx_session: int,
    pr_task_type: PRTaskType,
    signal_type: SignalType,
    idx_finger: int,
    idx_combination: int,
    force_direction: ForceDirection,
    idx_trial: int,
) -> tuple[str, str]:

    assert isinstance(dataset, Dataset)

    if dataset == Dataset.PR:

        assert signal_type != SignalType.FORCE
        assert idx_finger is None
        assert idx_combination is None
        assert force_direction is None

    elif dataset == Dataset.MVC:

        assert pr_task_type is None
        assert idx_combination is None
        assert idx_trial is None

    elif dataset == Dataset.ONEDOF:

        assert pr_task_type is None
        assert idx_combination is None
        assert force_direction is None

    elif dataset == Dataset.NDOF:

        assert pr_task_type is None
        assert idx_finger is None
        assert force_direction is None

    elif dataset == Dataset.RANDOM:

        assert pr_task_type is None
        assert idx_finger is None
        assert idx_combination is None
        assert force_direction is None

    else:
        raise NotImplementedError

    directory_fullpath = \
        HYSER_PATH_DOWNLOADED + \
        f"{dataset.value}_dataset" + \
        f"/" + \
        f"subject{idx_subject + 1 :02d}_" + \
        f"session{idx_session + 1}" + \
        f"/"

    if dataset == Dataset.PR:

        filename_noext = \
            f"{pr_task_type.value}_" + \
            f"{signal_type.value}_" + \
            f"sample{idx_trial + 1}"

    elif dataset == Dataset.MVC:

        filename_noext = \
            f"{dataset.value}_" + \
            f"{signal_type.value}_" + \
            f"finger{idx_finger + 1}_" + \
            f"{force_direction.value}"

    elif dataset == Dataset.ONEDOF:

        filename_noext = \
            f"{dataset.value}_" + \
            f"{signal_type.value}_" + \
            f"finger{idx_finger + 1}_" + \
            f"sample{idx_trial + 1}"

    elif dataset == Dataset.NDOF:

        filename_noext = \
            f"{dataset.value}_" + \
            f"{signal_type.value}_" + \
            f"combination{idx_combination + 1}_" + \
            f"sample{idx_trial + 1}"

    elif dataset == Dataset.RANDOM:

        filename_noext = \
            f"{dataset.value}_" + \
            f"{signal_type.value}_" + \
            f"sample{idx_trial + 1}"

    else:
        raise NotImplementedError

    file_fullpath_noext = directory_fullpath + filename_noext

    return file_fullpath_noext


def load_hdsemg(
    dataset: Dataset,
    idx_subject: int,
    idx_session: int,
    pr_task_type: PRTaskType,
    hdsemg_signal_type: SignalType.RAW | SignalType.PREPROCESS,  # only HD-sEMG
    idx_finger: int,
    idx_combination: int,
    force_direction: ForceDirection,
    idx_trial: int,
) -> np.ndarray[np.float32]:

    # The core is the function for loading:
    # https://wfdb.readthedocs.io/en/latest/wfdb.html?highlight=wfdb.rdsamp#wfdb.rdsamp

    assert isinstance(dataset, Dataset)
    assert idx_subject in range(NUM_SUBJECTS)
    assert idx_session in range(NUM_SESSIONS)
    if dataset == Dataset.PR:
        assert isinstance(pr_task_type, PRTaskType)
    else:
        assert pr_task_type is None
    assert hdsemg_signal_type in [SignalType.RAW, SignalType.PREPROCESS]
    if dataset in [Dataset.MVC, Dataset.ONEDOF]:
        assert idx_finger in range(NUM_FINGERS)
    else:
        assert idx_finger is None
    if dataset == Dataset.NDOF:
        assert idx_combination in range(NUM_COMBINATIONS_NDOF)
    else:
        assert idx_combination is None
    if dataset == Dataset.MVC:
        assert isinstance(force_direction, ForceDirection)
    else:
        assert force_direction is None
    # TODO: assert idx_trial

    filepath_noext = filepath_noext_from_indices(
        dataset=dataset,
        idx_subject=idx_subject,
        idx_session=idx_session,
        pr_task_type=pr_task_type,
        signal_type=hdsemg_signal_type,
        idx_finger=idx_finger,
        idx_combination=idx_combination,
        force_direction=force_direction,
        idx_trial=idx_trial,
    )

    hdsemg_v, fields = wfdb.rdsamp(filepath_noext)
    hdsemg_v = hdsemg_v.astype(np.float32)  # from float64 to float32
    hdsemg_v = hdsemg_v.T  # from (samples, ch) to (ch, samples)

    # TODO: assert fields

    return hdsemg_v


def load_force(
    dataset: Dataset,
    idx_subject: int,
    idx_session: int,
    idx_finger: int,
    idx_combination: int,
    force_direction: ForceDirection,
    idx_trial: int,
) -> np.ndarray[np.float32]:

    # The core is the function for loading:
    # https://wfdb.readthedocs.io/en/latest/wfdb.html?highlight=wfdb.rdsamp#wfdb.rdsamp

    assert isinstance(dataset, Dataset)
    assert idx_subject in range(NUM_SUBJECTS)
    assert idx_session in range(NUM_SESSIONS)
    if dataset in [Dataset.MVC, Dataset.ONEDOF]:
        assert idx_finger in range(NUM_FINGERS)
    else:
        assert idx_finger is None
    if dataset == Dataset.NDOF:
        assert idx_combination in range(NUM_COMBINATIONS_NDOF)
    else:
        assert idx_combination is None
    if dataset == Dataset.MVC:
        assert isinstance(force_direction, ForceDirection)
    else:
        assert force_direction is None
    # TODO: assert idx_trial

    filepath_noext = filepath_noext_from_indices(
        dataset=dataset,
        idx_subject=idx_subject,
        idx_session=idx_session,
        pr_task_type=None,  # <-- this is fixed for force
        signal_type=SignalType.FORCE,  # <-- this is fixed for force
        idx_finger=idx_finger,
        idx_combination=idx_combination,
        force_direction=force_direction,
        idx_trial=idx_trial,
    )

    force_v, fields = wfdb.rdsamp(filepath_noext)
    force_v = force_v.astype(np.float32)  # from float64 to float32
    force_v = force_v.T  # from (samples, ch) to (ch, samples)

    # TODO: assert fields

    return force_v


def load_hdsemg_and_force(
    dataset: Dataset,
    idx_subject: int,
    idx_session: int,
    pr_task_type: PRTaskType,
    hdsemg_signal_type: SignalType.RAW | SignalType.PREPROCESS,
    idx_finger: int,
    idx_combination: int,
    force_direction: ForceDirection,
    idx_trial: int,
) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:

    hdsemg_v = load_hdsemg(
        dataset=dataset,
        idx_subject=idx_subject,
        idx_session=idx_session,
        pr_task_type=pr_task_type,
        hdsemg_signal_type=hdsemg_signal_type,
        idx_finger=idx_finger,
        idx_combination=idx_combination,
        force_direction=force_direction,
        idx_trial=idx_trial,
    )

    force_v = load_force(
        dataset=dataset,
        idx_subject=idx_subject,
        idx_session=idx_session,
        idx_finger=idx_finger,
        idx_combination=idx_combination,
        force_direction=force_direction,
        idx_trial=idx_trial,
    )

    # subsample the HD-sEMG channels from 256 to 64: step 2 in both dimensions
    num_samples_hdsemg = hdsemg_v.shape[1]
    hdsemg_v = hdsemg_v.reshape((8, 8, 4, num_samples_hdsemg))
    hdsemg_v = hdsemg_v[::8, ::8]
    hdsemg_v = hdsemg_v.reshape((4, num_samples_hdsemg))

    # interpolate the force
    num_hi = hdsemg_v.shape[1]
    force_v_interp = np.zeros((NUM_CHANNELS_FORCE, num_hi), dtype=np.float32)
    for idx_hi in range(num_hi):
        idx_lo = idx_hi * FS_FORCE // FS_HDSEMG
        force_v_interp[:, idx_hi] = force_v[:, idx_lo]

    return hdsemg_v, force_v_interp


def concatenate_paired_segments(
    x: list[np.ndarray[np.float32]] | None,  # any hd-semg or feature
    f: list[np.ndarray[np.float32]] | None,  # force
) -> tuple[
    np.ndarray[np.float32] | None,
    np.ndarray[np.float32] | None,
]:

    # Wrapper of numpy.hstack in a paired fashion

    x = np.hstack(x) if x is not None else None
    f = np.hstack(f) if f is not None else None

    return x, f


# def resample_by_hold(
#    x: np.ndarray[np.float32],
#    fs_ratio: float,
# ) -> np.ndarray[np.float32]:
#
#    assert 0.0 <= fs_ratio <= 1.0
#
#    num_orig = x.shape[1]  # the large one
#    num_resamp = num_orig * fs_ratio  # the small one; not integer
#    ids_resamp = np.arange(num_resamp) / num_resamp * num_orig  # not integer
#    ids_resamp = np.floor(ids_resamp).astype(np.uint64)
#
#    x = x[:, ids_resamp]
#
#    return x
#
#
# def downsample_myo_as_force(
#    x: np.ndarray[np.float32],  # any hd-semg or feature, sampled at FS_HDSEMG
# ) -> np.ndarray[np.float32]:
#
#    fs_ratio = FS_FORCE / FS_HDSEMG
#    x = resample_by_hold(x, fs_ratio)
#
#    return x


def main() -> None:
    pass


if __name__ == '__main__':
    main()
