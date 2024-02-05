from __future__ import annotations
from itertools import product

from incremental_hyser.hyser import hyser as hy


def test_open_mvc() -> None:

    for idx_subj, idx_sess, idx_finger, emgsigtype, force_direction in product(
        range(hy.NUM_SUBJECTS),
        range(hy.NUM_SESSIONS),
        range(hy.NUM_FINGERS),
        [hy.SignalType.RAW, hy.SignalType.PREPROCESS],
        hy.ForceDirection,
        ):
        
        x, y = hy.load_hdsemg_and_force(
            dataset=hy.Dataset.MVC,
            idx_subject=idx_subj,
            idx_session=idx_sess,
            pr_task_type=None,
            hdsemg_signal_type=emgsigtype,
            idx_finger=idx_finger,
            idx_combination=None,
            force_direction=force_direction,
            idx_trial=None,
        )

        assert x.shape[0] == hy.NUM_CHANNELS_HDSEMG
        assert y.shape[0] == hy.NUM_CHANNELS_FORCE

        print(idx_subj, idx_sess, idx_finger, emgsigtype, force_direction)
    
    return


def test_open_1dof() -> None:

    for idx_subj, idx_sess, idx_finger, idx_trial, emgsigtype in product(
        range(hy.NUM_SUBJECTS),
        range(hy.NUM_SESSIONS),
        range(hy.NUM_FINGERS),
        range(hy.NUM_TRIALS_ONEDOF),
        [hy.SignalType.RAW, hy.SignalType.PREPROCESS],
        ):
        
        x, y = hy.load_hdsemg_and_force(
            dataset=hy.Dataset.ONEDOF,
            idx_subject=idx_subj,
            idx_session=idx_sess,
            pr_task_type=None,
            hdsemg_signal_type=emgsigtype,
            idx_finger=idx_finger,
            idx_combination=None,
            force_direction=None,
            idx_trial=idx_trial,
        )

        assert x.shape[0] == hy.NUM_CHANNELS_HDSEMG
        assert y.shape[0] == hy.NUM_CHANNELS_FORCE

        print(idx_subj, idx_sess, idx_finger, idx_trial, emgsigtype)

    return


def test_open_ndof() -> None:

    for idx_subj, idx_sess, idx_combination, idx_trial, emgsigtype in product(
        range(hy.NUM_SUBJECTS),
        range(hy.NUM_SESSIONS),
        range(hy.NUM_COMBINATIONS_NDOF),
        range(hy.NUM_TRIALS_NDOF),
        [hy.SignalType.RAW, hy.SignalType.PREPROCESS],
        ):
        
        x, y = hy.load_hdsemg_and_force(
            dataset=hy.Dataset.NDOF,
            idx_subject=idx_subj,
            idx_session=idx_sess,
            pr_task_type=None,
            hdsemg_signal_type=emgsigtype,
            idx_finger=None,
            idx_combination=idx_combination,
            force_direction=None,
            idx_trial=idx_trial,
        )

        assert x.shape[0] == hy.NUM_CHANNELS_HDSEMG
        assert y.shape[0] == hy.NUM_CHANNELS_FORCE

        print(idx_subj, idx_sess, idx_combination, idx_trial, emgsigtype)

    return


def test_open_random() -> None:

    for idx_subj, idx_sess, idx_trial, emgsigtype in product(
        range(hy.NUM_SUBJECTS),
        range(hy.NUM_SESSIONS),
        range(hy.NUM_TRIALS_NDOF),
        [hy.SignalType.RAW, hy.SignalType.PREPROCESS],
        ):
        
        x, y = hy.load_hdsemg_and_force(
            dataset=hy.Dataset.RANDOM,
            idx_subject=idx_subj,
            idx_session=idx_sess,
            pr_task_type=None,
            hdsemg_signal_type=emgsigtype,
            idx_finger=None,
            idx_combination=None,
            force_direction=None,
            idx_trial=idx_trial,
        )

        assert x.shape[0] == hy.NUM_CHANNELS_HDSEMG
        assert y.shape[0] == hy.NUM_CHANNELS_FORCE

        print(idx_subj, idx_sess, idx_trial, emgsigtype)

    return


def main() -> None:
    
    print("\n\n\n\n\nMVC\n\n\n\n\n")
    test_open_mvc()
    
    print("\n\n\n\n\n1-DOF\n\n\n\n\n")
    test_open_1dof()
    
    print("\n\n\n\n\nN-DOF\n\n\n\n\n")
    test_open_ndof()
    
    print("\n\n\n\n\nRANDOM\n\n\n\n\n")
    test_open_random()

    return


if __name__ == "__main__":
    main()
