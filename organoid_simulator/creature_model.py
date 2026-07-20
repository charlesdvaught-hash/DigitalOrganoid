"""
Offline engine replicating the spiking model used in organoid_creature (index.html):
Izhikevich neurons (RS/IB excitatory, FS/LTS inhibitory), Dale's law, STDP,
homeostatic-style metabolism (energy/fatigue), sparse K-out connectivity.

This intentionally mirrors organoid_simulator/model.py's role (build network,
run dynamics) but for the spiking/STDP/metabolism model instead of the
tanh-reservoir model. No arena/body/3D migration here — driven by random
input, same abstraction sweep.py already uses for the reservoir model.
"""
import numpy as np

SUBTYPE_RS, SUBTYPE_IB, SUBTYPE_FS, SUBTYPE_LTS = 0, 1, 2, 3

# STDP / plasticity constants (from index.html)
TRACE_DECAY = 0.96
A_PLUS = 0.008
A_MINUS = 0.010
W_MAX = 2.5


def build_creature_network(N, K, fi=0.2, seed=None):
    """
    Build N Izhikevich neurons (excitatory/inhibitory split by fi) with
    subtype params jittered +/-8%, plus a sparse K-out synapse list with
    Dale-consistent signed weights. Mirrors index.html's init loop.
    """
    rng = np.random.default_rng(seed)

    types = np.where(rng.random(N) < fi, -1, 1).astype(np.int8)
    subtypes = np.zeros(N, dtype=np.int8)
    a = np.zeros(N); b = np.zeros(N); c = np.zeros(N); d = np.zeros(N)

    for i in range(N):
        if types[i] > 0:
            if rng.random() < 0.85:
                subtypes[i] = SUBTYPE_RS
                base = (0.02, 0.2, -65, 8)
            else:
                subtypes[i] = SUBTYPE_IB
                base = (0.02, 0.2, -50, 2)
        else:
            if rng.random() < 0.80:
                subtypes[i] = SUBTYPE_FS
                base = (0.1, 0.2, -65, 2)
            else:
                subtypes[i] = SUBTYPE_LTS
                base = (0.02, 0.25, -65, 2)

        jitter = lambda: 1.0 + (rng.random() * 2 - 1) * 0.08
        a[i] = base[0] * jitter()
        b[i] = base[1] * jitter()
        c[i] = base[2] + (rng.random() * 2 - 1) * 2.0
        d[i] = base[3] * jitter()

    v = c.copy()
    u = b * v

    # Sparse K-out synapse list, Dale-signed by presynaptic type, half-normal magnitude.
    pre = np.repeat(np.arange(N), K)
    post = np.empty(N * K, dtype=np.int64)
    for i in range(N):
        candidates = rng.choice(np.delete(np.arange(N), i), size=K, replace=False)
        post[i * K:(i + 1) * K] = candidates
    mag = np.abs(rng.standard_normal(N * K))
    weight = mag * types[pre]  # signed by presynaptic (Dale-consistent) identity

    return dict(N=N, K=K, types=types, subtypes=subtypes, a=a, b=b, c=c, d=d,
                v=v, u=u, pre=pre, post=post, weight=weight.astype(np.float64))


def run_creature_simulation(net, T, dale=True, stdp_on=True, noise_scale=0.15,
                             seed=None, record_every=1):
    """
    Vectorized version of stepBrain(): Izhikevich update + Dale-gated synaptic
    current + metabolism (energy/fatigue) + optional STDP weight updates.
    Driven by random sensor-style input (same convention as the reservoir
    sweep.py, which drives with i.i.d. uniform noise rather than embodiment).

    Returns dict with spike raster (T, N) as booleans, weight trajectory
    summary, and final energy/fatigue.
    """
    rng = np.random.default_rng(seed)
    N = net['N']
    v, u = net['v'].copy(), net['u'].copy()
    a, b, c, d = net['a'], net['b'], net['c'], net['d']
    pre, post, weight = net['pre'], net['post'], net['weight'].copy()
    types = net['types']

    energy = np.ones(N)
    fatigue = np.zeros(N)
    trace_pre = np.zeros(N)
    trace_post = np.zeros(N)
    smoothed_activity = np.zeros(N)

    n_rec = (T + record_every - 1) // record_every
    spikes = np.zeros((n_rec, N), dtype=bool)
    weight_history = np.zeros(n_rec)
    rec_idx = 0

    for t in range(T):
        trace_pre *= TRACE_DECAY
        trace_post *= TRACE_DECAY

        # Recovery equations from index.html: energy[i] += (1 - energy[i]) * 0.001 * recoveryRate
        energy += (1.0 - energy) * 0.001
        fatigue *= 0.98
        excitability = energy * (1.0 - fatigue * 0.5)

        # Synaptic current: Dale ON uses signed weight, Dale OFF uses |weight|.
        # We drive it using smoothed_activity[pre] instead of tanh(v[pre]/30.0).
        syn_w = weight if dale else np.abs(weight)
        I_syn = np.zeros(N)
        np.add.at(I_syn, post, syn_w * smoothed_activity[pre])
        I_effective = I_syn * excitability

        I_noise = rng.standard_normal(N) * noise_scale
        rare_kick = rng.random(N) < 0.002
        I_noise = np.where(rare_kick, I_noise + 15.0, I_noise)

        I = I_effective + I_noise

        # Fatigue throttle: I *= (1.0 - (fatigue[i] - 0.6) * 0.5) when fatigue[i] > 0.6
        throttle_mask = fatigue > 0.6
        if throttle_mask.any():
            I[throttle_mask] *= (1.0 - (fatigue[throttle_mask] - 0.6) * 0.5)

        # Izhikevich update (2 sub-steps for stability, per index.html).
        for _ in range(2):
            v += 0.5 * (0.04 * v * v + 5 * v + 140 - u + I)
            u += 0.5 * a * (b * v - u)

        # Stability limits from index.html
        np.clip(v, -90.0, 35.0, out=v)

        fired = v >= 30.0
        # Update smoothed_activity: smoothedActivity[i] = smoothedActivity[i] * 0.8 + isSpike * 0.2
        is_spike = fired.astype(np.float64)
        smoothed_activity = smoothed_activity * 0.8 + is_spike * 0.2

        if fired.any():
            v[fired] = c[fired]
            u[fired] += d[fired]
            fatigue[fired] = np.minimum(1.0, fatigue[fired] + 0.05)
            energy[fired] = np.maximum(0.0, energy[fired] - 0.002)
            trace_post[fired] += 1.0
            trace_pre[fired] += 1.0

            if stdp_on:
                fired_idx = np.flatnonzero(fired)
                pre_fired_mask = np.isin(pre, fired_idx)
                if pre_fired_mask.any():
                    weight[pre_fired_mask] += A_PLUS * trace_post[post[pre_fired_mask]]
                post_fired_mask = np.isin(post, fired_idx)
                if post_fired_mask.any():
                    weight[post_fired_mask] -= A_MINUS * trace_pre[pre[post_fired_mask]]

                # Dale's law enforcement on weights post-STDP.
                # Excitatory (types > 0) must stay positive/zero.
                # Inhibitory (types < 0) must stay negative/zero.
                # W_MAX limit
                exc_syn_mask = types[pre] > 0
                inh_syn_mask = types[pre] < 0
                weight[exc_syn_mask] = np.clip(weight[exc_syn_mask], 0.0, W_MAX)
                weight[inh_syn_mask] = np.clip(weight[inh_syn_mask], -W_MAX, 0.0)

        energy = np.clip(energy, 0.0, 1.0)
        fatigue = np.clip(fatigue, 0.0, 1.0)

        if t % record_every == 0:
            spikes[rec_idx] = fired
            weight_history[rec_idx] = np.mean(np.abs(weight))
            rec_idx += 1

    return dict(spikes=spikes[:rec_idx], weight_history=weight_history[:rec_idx],
                final_energy=energy, final_fatigue=fatigue, final_weight=weight)
