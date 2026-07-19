"""
Metrics for the creature's spiking output. Analog to organoid_simulator/metrics.py
(which measures reservoir memory capacity + avalanche criticality on continuous
states) but adapted to binary spike rasters.
"""
import numpy as np


def firing_stats(spikes):
    """spikes: (T, N) bool. Returns mean rate, per-neuron rate std, and CV of
    the population rate (a simple synchrony proxy: higher CV = burstier/more
    synchronized population activity)."""
    pop_rate = spikes.mean(axis=1)  # fraction of N firing each tick
    mean_rate = float(pop_rate.mean())
    rate_std = float(pop_rate.std())
    cv = float(rate_std / mean_rate) if mean_rate > 0 else float('nan')
    per_neuron_rate = spikes.mean(axis=0)
    return dict(mean_pop_rate=mean_rate, pop_rate_cv=cv,
                per_neuron_rate_std=float(per_neuron_rate.std()))


def avalanche_stats(spikes, bin_size=1):
    """Simple avalanche size distribution: contiguous runs of population
    activity above zero, binned. Returns count and mean/size std as a coarse
    criticality proxy (no power-law fit dependency, unlike metrics.py)."""
    pop_count = spikes.sum(axis=1)
    if bin_size > 1:
        n = len(pop_count) - (len(pop_count) % bin_size)
        pop_count = pop_count[:n].reshape(-1, bin_size).sum(axis=1)

    active = pop_count > 0
    sizes = []
    run = 0
    for a in active:
        if a:
            run += 1
        elif run > 0:
            sizes.append(run)
            run = 0
    if run > 0:
        sizes.append(run)

    sizes = np.array(sizes) if sizes else np.array([0])
    return dict(num_avalanches=len(sizes), mean_avalanche_size=float(sizes.mean()),
                max_avalanche_size=int(sizes.max()))


def weight_drift(net, result):
    """Mean |weight| change from init to final — how much STDP moved the
    network from its random init."""
    init_mean = float(np.mean(np.abs(net['weight'])))
    final_mean = float(np.mean(np.abs(result['final_weight'])))
    return dict(init_mean_abs_weight=init_mean, final_mean_abs_weight=final_mean,
                weight_drift=final_mean - init_mean)


def metabolic_stats(result):
    return dict(mean_final_energy=float(result['final_energy'].mean()),
                mean_final_fatigue=float(result['final_fatigue'].mean()))
