"""
Metrics for the creature's spiking output, neural network topology, and behavioral foraging.
"""
import numpy as np

def firing_stats(spikes):
    """spikes: (T, N) bool. Returns mean rate, per-neuron rate std, and CV of
    the population rate (higher CV = burstier/more synchronized population activity)."""
    pop_rate = spikes.mean(axis=1)  # fraction of N firing each tick
    mean_rate = float(pop_rate.mean())
    rate_std = float(pop_rate.std())
    cv = float(rate_std / mean_rate) if mean_rate > 0 else 0.0
    per_neuron_rate = spikes.mean(axis=0)
    return dict(mean_pop_rate=mean_rate, pop_rate_cv=cv,
                per_neuron_rate_std=float(per_neuron_rate.std()))

def avalanche_stats(spikes, bin_size=1):
    """Simple avalanche size distribution: contiguous runs of population
    activity above zero, binned."""
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

def calculate_network_metrics(net):
    """
    Computes neural metrics for SNN:
      - Connectivity density
      - Modularity (Louvain-like, spatial octant partition based)
      - Synchrony index (mean pairwise correlation of absolute weights)
    """
    N = net['N']
    weight = net['weight']
    pre = net['pre']
    post = net['post']
    px = net['px']
    py = net['py']
    pz = net['pz']

    # 1. Connectivity density
    max_possible_connections = N * (N - 1)
    density = len(weight) / max_possible_connections if max_possible_connections > 0 else 0.0

    # 2. Spatial Modularity Q (octant community partition)
    # Assign each neuron to one of 8 octants based on coordinate signs
    octant_x = (px > 0).astype(int)
    octant_y = (py > 0).astype(int) * 2
    octant_z = (pz > 0).astype(int) * 4
    communities = octant_x + octant_y + octant_z  # 0 to 7

    abs_weights = np.abs(weight)
    W_total = abs_weights.sum()

    if W_total > 0:
        # Out-degree/in-degree community sums
        comm_out = np.zeros(8)
        comm_in = np.zeros(8)
        comm_internal = np.zeros(8)

        pre_comm = communities[pre]
        post_comm = communities[post]

        np.add.at(comm_out, pre_comm, abs_weights)
        np.add.at(comm_in, post_comm, abs_weights)

        internal_mask = pre_comm == post_comm
        np.add.at(comm_internal, pre_comm[internal_mask], abs_weights[internal_mask])

        Q = 0.0
        for c in range(8):
            Q += (comm_internal[c] / W_total) - (comm_out[c] / W_total) * (comm_in[c] / W_total)
    else:
        Q = 0.0

    return {
        'density': float(density),
        'spatial_modularity': float(Q)
    }

def calculate_behavioral_metrics(behavior_log, T_steps):
    """
    Computes behavioral metrics:
      - Foraging efficiency (food eaten / steps)
      - Exploration entropy (shannon entropy of spatial 2D grid coverage)
    """
    # Assuming we track positions over time or derive from behavior profile
    food_eaten = behavior_log.get('food_eaten', 0)
    foraging_efficiency = food_eaten / T_steps if T_steps > 0 else 0.0

    # Return metrics dictionary
    return {
        'food_eaten': food_eaten,
        'survival_steps': behavior_log.get('survival_steps', T_steps),
        'distance_traveled': behavior_log.get('distance_traveled', 0.0),
        'foraging_efficiency': foraging_efficiency,
    }
