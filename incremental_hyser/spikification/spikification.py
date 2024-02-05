from __future__ import annotations

import numpy as np
# import scipy.signal as ssg

# from ..hyser import hyser as hy
from . import lif


def spikify(
    hdsemg_v: np.ndarray[np.float32],
    fs_hz: float,
    # lowcut_hz: float,
    # highcut_hz: float,
    # order: int,
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
) -> tuple[
    # np.ndarray[np.float32],  # hdsemg_bp_v  # TOGLIERE ???
    np.ndarray[np.float32],  # x_pre
    np.ndarray[np.float32],  # spike_times_s
    np.ndarray[np.uint32],  # spike_neuron_ids
    np.ndarray[np.float32],  # x_post
]:

    assert len(hdsemg_v.shape) == 2

    # bandpass and rectify
    # sos = create_butterworth_filter(fs_hz, lowcut_hz, highcut_hz, order)
    # hdsemg_bp_v = ssg.sosfilt(sos, hdsemg_v)
    # del hdsemg_v
    # hdsemg_rect_v = np.abs(hdsemg_bp_v)  # full-wave rectification
    hdsemg_rect_v = np.abs(hdsemg_v)  # full-wave rectification

    # ----------------------------------------------------------------------- #
    # TODO: code these better
    # ##dt_sim_s = 1.0 / hy.FS_HDSEMG
    # ##monitors_dt_s = 1.0 / hy.FS_HDSEMG  # pre-synaptic
    # ##monitor_dt_s = 1.0 / hy.FS_HDSEMG  # post-synaptic
    # ##report = 'stdout'
    # ##time_total_s = hdsemg_rect_v.shape[1] / hy.FS_HDSEMG
    # ----------------------------------------------------------------------- #

    num_samples = hdsemg_rect_v.shape[1]
    time_total_s = num_samples / fs_hz  # needed by the post-synaptic

    x_drive = hdsemg_rect_v * gain_per_volt
    del hdsemg_rect_v

    # pre-synaptic

    x_pre, spike_times_s, spike_neuron_ids = lif.lif_presynaptic(
        x_drive=x_drive,
        fs_hz=fs_hz,
        dt_sim_s=dt_sim_s,
        dt_monitors_s=dt_monitors_pre_s,
        x_init=x_init,
        x_reset=x_reset,
        tau_s=taupre_s,
        trefr_s=trefr_s,
        report_stdstream=report_stdstream,
        report_period_s=report_period_s,
    )
    del x_drive

    # post-synaptic

    x_post = lif.lif_postsynaptic(
        inspike_times_s=spike_times_s,
        inspike_neuron_ids=spike_neuron_ids,
        time_total_s=time_total_s,
        dt_sim_s=dt_sim_s,
        dt_monitor_s=dt_monitor_post_s,
        tau_s=taupost_s,
        report_stdstream=report_stdstream,
        report_period_s=report_period_s,
    )

    x_post = lif.normalize_xpost(x_post, trefr_s, taupost_s)

    return x_pre, spike_times_s, spike_neuron_ids, x_post


def main():
    pass


if __name__ == '__main__':
    main()
