from __future__ import annotations

import numpy as np
import brian2 as b2

from ..hyser import hyser as hy


NUM_NEURONS = hy.NUM_CHANNELS_HDSEMG


def lif_presynaptic(
    x_drive: np.ndarray,
    fs_hz: float,
    x_init: float,
    x_reset: float,
    tau_s: float,
    trefr_s: float,
    dt_sim_s: float,
    dt_monitors_s: float,
    report_stdstream: str | None,
    report_period_s: float | None,
) -> tuple[
    np.ndarray[np.float32],  # x
    np.ndarray[np.float32],  # spike_times_s
    np.ndarray[np.uint8],  # spike_neuron_ids (NB: uint8)
]:

    b2.start_scope()

    # convert all time and frequency arguments to Brian2 physical quantities
    fs = fs_hz * b2.hertz
    tau = tau_s * b2.second
    trefr = trefr_s * b2.second
    dt_sim = dt_sim_s * b2.second
    dt_monitors = dt_monitors_s * b2.second
    if report_period_s is None:
        report_period_s = 10.0  # Brian2's default
    report_period = report_period_s * b2.second
    del fs_hz, tau_s, trefr_s, dt_sim_s, dt_monitors_s, report_period_s

    assert len(x_drive.shape) == 2
    assert x_drive.shape[0] == NUM_NEURONS
    num_samples = x_drive.shape[1]
    time_total = num_samples / fs
    dt_sample = 1.0 / fs
    x_drive = b2.TimedArray(values=x_drive.T, dt=dt_sample)  # transposed!

    eqn = \
        '''
        dx/dt = (-x + x_drive(t, i)) / tau : 1 (unless refractory)
        '''

    neurongroup = b2.NeuronGroup(
        N=NUM_NEURONS,
        model=eqn,
        method='exact',
        threshold='x > 1',
        reset='x = x_reset',
        refractory=trefr,
        dtype=np.float32,
        dt=dt_sim,
    )

    neurongroup.x = x_init

    statemonitor = b2.StateMonitor(
        source=neurongroup,
        variables='x',
        record=True,  # "True" monitors all indices
        dt=dt_monitors,
    )

    # spike monitor
    spikemonitor = b2.SpikeMonitor(
        source=neurongroup,
        variables=None,  # none in addition to timestamp and spiker's index
        record=True,  # "True" monitors all indices
    )

    # (the net is created just for the scheduling summary)
    network = b2.Network(neurongroup, statemonitor, spikemonitor)
    scheduling_summary = network.scheduling_summary()
    del network
    print(f"\n\n{scheduling_summary}\n\n")

    b2.run(
        duration=time_total,
        report=report_stdstream,
        report_period=report_period,
    )

    x = statemonitor.x
    spike_times_s = np.float32(spikemonitor.t / b2.second)
    spike_neuron_ids = np.uint16(spikemonitor.i)
    # check whether sorted, and sort if needed
    # (I do not know if they are created sorted by design from the run; it
    # looks so, but I enforce it to be sure.)
    nonproper_isis_s = np.diff(spike_times_s)
    sorted = np.all(nonproper_isis_s >= 0.0)
    if not sorted:
        sort_idxs = np.argsort(spike_times_s)
        spike_times_s = spike_times_s[sort_idxs]
        spike_neuron_ids = spike_neuron_ids[sort_idxs]

    return x, spike_times_s, spike_neuron_ids


def lif_postsynaptic(
    inspike_times_s: np.ndarray[np.float32],
    inspike_neuron_ids: np.ndarray[np.uint8],
    tau_s: float,
    time_total_s: float,
    dt_sim_s: float,
    dt_monitor_s: float,
    report_stdstream: str | None,
    report_period_s: float | None,
) -> np.ndarray[np.float32]:

    # sanity check: same number of spikes and spikers
    num_spikes = len(inspike_times_s)
    assert len(inspike_neuron_ids) == num_spikes, \
        "Spikers' indices must be as many as the spikes' timestamps!"

    # check whether sorted, and sort if needed
    nonproper_isis_s = np.diff(inspike_times_s)
    sorted = np.all(nonproper_isis_s >= 0.0)
    if not sorted:
        sort_idxs = np.argsort(inspike_times_s)
        inspike_times_s = inspike_times_s[sort_idxs]
        inspike_neuron_ids = inspike_neuron_ids[sort_idxs]

    b2.start_scope()

    # convert all time arguments to Brian2 physical quantities
    inspike_times = inspike_times_s * b2.second
    tau = tau_s * b2.second
    time_total = time_total_s * b2.second
    dt_sim = dt_sim_s * b2.second
    dt_monitor = dt_monitor_s * b2.second
    if report_period_s is None:
        report_period_s = 10.0  # Brian2's default
    report_period = report_period_s * b2.second
    del (
        inspike_times_s, tau_s, time_total_s,
        dt_sim_s, dt_monitor_s, report_period_s,
    )

    spikegengroup = b2.SpikeGeneratorGroup(
        N=NUM_NEURONS,
        indices=inspike_neuron_ids,
        times=inspike_times,
        dt=dt_sim,
        when='thresholds',
        sorted=True,  # because they have just been sorted prior to this
    )

    eqn = \
        '''
        dx/dt = - x / tau : 1
        '''

    neurongroup = b2.NeuronGroup(
        N=NUM_NEURONS,
        model=eqn,
        method='exact',
        dtype=np.float32,
        dt=dt_sim,
    )

    synapses = b2.Synapses(
        source=spikegengroup,
        target=neurongroup,
        on_pre='x += 1.0',
        dtype=np.float32,
        dt=dt_sim,
    )
    synapses.connect(j='i')

    neurongroup.x = 0.0

    statemonitor = b2.StateMonitor(
        source=neurongroup,
        variables='x',
        record=True,  # "True" monitors all indices
        dt=dt_monitor,
        when='end',
    )

    # (the net is created just for the scheduling summary)
    network = b2.Network(neurongroup, synapses, statemonitor)
    scheduling_summary = network.scheduling_summary()
    del network
    print(f"\n\n{scheduling_summary}\n\n")

    b2.run(
        duration=time_total,
        report=report_stdstream,
        report_period=report_period,
    )

    x = statemonitor.x
    del statemonitor
    # x = x[:, 1:]  # discard the first because it is time 0.0  # TODO: NEEDED?

    return x


def normalize_xpost(
    xpost: np.ndarray[np.float32],
    trefr_s: float,
    taupost_s: float,
) -> np.ndarray[np.float32]:

    assert (xpost >= 0.0).all()
    assert trefr_s >= 0.0
    assert taupost_s > 0.0  # for the series to converge

    # mathematically maximum value, used for final normalization
    xpost_max_analytical = 1.0 / (1.0 - np.exp(- trefr_s / taupost_s))

    # normalization
    xpost_norm = xpost / xpost_max_analytical
    xpost_norm = xpost_norm.clip(max=1.0)

    return xpost_norm


def main():
    pass


if __name__ == '__main__':
    main()
