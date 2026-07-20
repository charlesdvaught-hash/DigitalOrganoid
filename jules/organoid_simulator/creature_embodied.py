"""
Embodied offline port of the creature closed loop in index.html.

Unlike creature_model.py (which drives a spiking net with random noise and has
"No arena/body/3D migration"), this module reproduces the actual embodied loop:

    sensors() -> 6 sensory pools -> stepBrain() -> motor pools -> differential
    drive body in a 2D arena with food -> energy / fatigue / U-shaped
    cost-of-transport -> eating -> dopamine -> reward-modulated STDP.

Faithful to index.html:
  - Izhikevich RS/IB/FS/LTS neurons, +/-8% param jitter, Dale's law.
  - Distance-based K-nearest connectivity on a unit-ball embedding, exp(-d^2/0.6)
    acceptance with a long-range escape prob, incoming-weight normalization.
  - Innate reflex links sL->mL, sR->mR, sW->mL/mR (excluded from plasticity).
  - Six sensory pools sL,sR,sW,sD,sH,sT injected as current (gain Sg*14).
  - Two chemosensory antennae (THETA_OFF, SMELL_SCALE) over ALL food; wall sense;
    graded food-distance sense; interoceptive hunger and fatigue.
  - Differential-drive body: speed from total motor firing, turn from L-R
    difference + proportional wall avoidance, heading inertia.
  - Metabolism: energyVal -= 0.030 + 2.0*speed + 400*speed^2 (U-shaped cost of
    transport); body tiredness accrues with movement; eating gives +28 energy,
    eases tiredness, and a dopamine burst.
  - Reward-modulated STDP: dw scaled by (0.1 + dopamine*4.0) when learning is on,
    gated by local energy; homeostatic synaptic scaling every 500 steps.

Deliberately excluded (documented simplifications relative to index.html):
  - Structural plasticity: synaptic pruning, reward-triggered synapse growth,
    and morphogen/migration self-organization. Topology is FIXED so the loop
    vectorizes and sweeps at speed. Synaptic (weight) plasticity is fully kept.
"""
import numpy as np

SUBTYPE_RS, SUBTYPE_IB, SUBTYPE_FS, SUBTYPE_LTS = 0, 1, 2, 3

# Plasticity constants (index.html)
TRACE_DECAY = 0.96
A_PLUS = 0.008
A_MINUS = 0.010
W_MAX = 2.5

# Fixed pool layout (index.html). Requires N > 190.
POOLS = {
    'sL': (0, 20), 'sR': (20, 40), 'sW': (40, 60),
    'sD': (60, 80), 'sH': (80, 100), 'sT': (100, 120),
    'mL': (120, 155), 'mR': (155, 190),
}

# Body / sensor constants (index.html)
THETA_OFF = 0.6
SMELL_SCALE = 0.35
SENSOR_GAIN = 1.5   # Sg default (slider 15 -> /10)
MOTOR_GAIN = 1.0    # Mg default (slider 10 -> /10)
BASE_NOISE = 0.01
DEV_STAGE = 0.20    # -> stage; devNoise = (1-stage)*0.015
LONG_RANGE_P = 0.10
DECAY_SCALE = 0.6   # non-cortical
TARGET_G = 1.4      # non-cortical norm target
HOMEO_STRENGTH = 1.0
EAT_RADIUS2 = 0.0009
N_FOOD = 6


def _angwrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def build_creature_network(N, K, fi=0.2, seed=None):
    """Distance-based connectivity + innate reflex links, matching index.html build()."""
    rng = np.random.default_rng(seed)

    # Unit-ball embedding (rejection sampling), as in build().
    px = np.empty(N); py = np.empty(N); pz = np.empty(N)
    filled = 0
    while filled < N:
        cand = rng.random((N, 3)) * 2 - 1
        ok = (cand ** 2).sum(1) <= 1.0
        take = cand[ok]
        n = min(len(take), N - filled)
        px[filled:filled+n] = take[:n, 0]
        py[filled:filled+n] = take[:n, 1]
        pz[filled:filled+n] = take[:n, 2]
        filled += n

    types = np.where(rng.random(N) < fi, -1, 1).astype(np.int8)
    # Pool neurons are forced excitatory.
    for (lo, hi) in POOLS.values():
        types[lo:hi] = 1

    subtypes = np.zeros(N, dtype=np.int8)
    a = np.zeros(N); b = np.zeros(N); c = np.zeros(N); d = np.zeros(N)
    for i in range(N):
        if types[i] > 0:
            if rng.random() < 0.85:
                subtypes[i] = SUBTYPE_RS; base = (0.02, 0.2, -65, 8)
            else:
                subtypes[i] = SUBTYPE_IB; base = (0.02, 0.2, -50, 2)
        else:
            if rng.random() < 0.80:
                subtypes[i] = SUBTYPE_FS; base = (0.1, 0.2, -65, 2)
            else:
                subtypes[i] = SUBTYPE_LTS; base = (0.02, 0.25, -65, 2)
        jit = lambda: 1.0 + (rng.random() * 2 - 1) * 0.08
        a[i] = base[0] * jit()
        b[i] = base[1] * jit()
        c[i] = base[2] + (rng.random() * 2 - 1) * 2.0
        d[i] = base[3] * jit()

    v = c.copy()
    u = b * v

    # Distance-based K-nearest connectivity (accuracy over speed, as index.html).
    pre_l = []; post_l = []; w_l = []
    coords = np.stack([px, py, pz], axis=1)
    for j in range(N):
        dd = ((coords - coords[j]) ** 2).sum(1)
        dd[j] = np.inf
        order = np.argsort(dd)
        found = 0
        for k in order:
            if found >= K:
                break
            prob = np.exp(-dd[k] / DECAY_SCALE)
            if rng.random() < LONG_RANGE_P:
                prob = 1.0
            if rng.random() < prob:
                w0 = abs(rng.standard_normal()) * types[k] * 0.8
                pre_l.append(int(k)); post_l.append(j); w_l.append(w0)
                found += 1

    pre = np.array(pre_l, dtype=np.int64)
    post = np.array(post_l, dtype=np.int64)
    weight = np.array(w_l, dtype=np.float64)
    is_reflex = np.zeros(len(pre), dtype=bool)

    # Incoming-weight normalization to TARGET_G (before reflex links, per build()).
    absum = np.zeros(N)
    np.add.at(absum, post, np.abs(weight))
    g = np.ones(N)
    nz = absum > 0
    g[nz] = TARGET_G / absum[nz]
    weight *= g[post]

    # Innate reflex links (excluded from plasticity), added after normalization.
    def link(sp, mp, wt):
        s0, s1 = POOLS[sp]; t0, t1 = POOLS[mp]
        for t in range(t0, t1):
            for s in range(s0, s1, 3):
                pre_l2.append(s); post_l2.append(t); w_l2.append(wt * types[s]); rfx.append(True)

    pre_l2 = []; post_l2 = []; w_l2 = []; rfx = []
    link('sL', 'mL', 0.6)
    link('sR', 'mR', 0.6)
    link('sW', 'mL', 0.4); link('sW', 'mR', 0.4)

    if pre_l2:
        pre = np.concatenate([pre, np.array(pre_l2, dtype=np.int64)])
        post = np.concatenate([post, np.array(post_l2, dtype=np.int64)])
        weight = np.concatenate([weight, np.array(w_l2, dtype=np.float64)])
        is_reflex = np.concatenate([is_reflex, np.array(rfx, dtype=bool)])

    return dict(N=N, K=K, types=types, subtypes=subtypes, a=a, b=b, c=c, d=d,
                v=v, u=u, pre=pre, post=post, weight=weight, is_reflex=is_reflex,
                px=px, py=py, pz=pz)


def _sensors(cx, cy, head, food, rng):
    """Two antennae + wall + food-distance, matching index.html sensors()."""
    if len(food) > 0:
        dx = food[:, 0] - cx; dy = food[:, 1] - cy
        dist = np.sqrt(dx * dx + dy * dy)
        nearest = float(dist.min())
        r = dist / SMELL_SCALE
        intensity = 1.0 / (1.0 + r * r)
        ang = _angwrap(np.arctan2(dy, dx) - head)
        L = float((intensity * np.maximum(0.0, np.cos(ang - THETA_OFF))).sum())
        R = float((intensity * np.maximum(0.0, np.cos(ang + THETA_OFF))).sum())
    else:
        nearest = 1e9; L = 0.0; R = 0.0
    L = 1.0 - np.exp(-L); R = 1.0 - np.exp(-R)
    L = min(1.0, max(0.0, L + (rng.random() - 0.5) * 0.04))
    R = min(1.0, max(0.0, R + (rng.random() - 0.5) * 0.04))
    wall = max(0.0, 1.0 - min(cx, cy, 1.0 - cx, 1.0 - cy) / 0.15)
    food_close = (1.0 / (1.0 + (nearest / 0.3) ** 2)) if len(food) > 0 else 0.0
    return L, R, wall, food_close, nearest


def run_creature_simulation(net, T, dale=True, stdp_on=True, seed=None, record_every=1):
    """Embodied closed loop. stdp_on gates reward-modulated plasticity (dev+reward)."""
    rng = np.random.default_rng(seed)
    N = net['N']
    v, u = net['v'].copy(), net['u'].copy()
    a, b, c, d = net['a'], net['b'], net['c'], net['d']
    pre, post = net['pre'], net['post']
    weight = net['weight'].copy()
    is_reflex = net['is_reflex']
    plastic = ~is_reflex
    types = net['types']

    energy = np.ones(N)          # per-neuron metabolic energy (0..1)
    fatigue = np.zeros(N)
    trace_pre = np.zeros(N)
    trace_post = np.zeros(N)
    smoothed = np.zeros(N)
    firing_rate = np.zeros(N)

    exc_syn = types[pre] > 0
    inh_syn = types[pre] < 0

    stage = DEV_STAGE
    noise_scale = BASE_NOISE + (1.0 - stage) * 0.015
    rare_p = 0.002 * (1.0 - stage)
    sensor_factor = SENSOR_GAIN * 14.0

    mL0, mL1 = POOLS['mL']; mR0, mR1 = POOLS['mR']

    # ---- body state ----
    cx, cy, head, head_vel = 0.5, 0.5, float(rng.random() * 6.28), 0.0
    energy_val = 100.0
    tiredness = 0.0
    score = 0
    dopamine = 0.0
    alive = True
    survival_steps = 0
    distance = 0.0
    speed_sum = 0.0
    food = np.column_stack([rng.random(N_FOOD) * 0.9 + 0.05,
                            rng.random(N_FOOD) * 0.9 + 0.05])

    n_rec = (T + record_every - 1) // record_every
    spikes = np.zeros((n_rec, N), dtype=bool)
    weight_history = np.zeros(n_rec)
    rec_idx = 0

    for t in range(T):
        # ---- sensing -> injection ----
        sensor_inj = np.zeros(N)
        if alive:
            L, R, W, food_close, nearest = _sensors(cx, cy, head, food, rng)
            hunger = 1.0 - energy_val / 100.0
            tired = min(1.0, tiredness)
            sensor_inj[0:20] = L * sensor_factor
            sensor_inj[20:40] = R * sensor_factor
            sensor_inj[40:60] = W * sensor_factor
            sensor_inj[60:80] = food_close * sensor_factor
            sensor_inj[80:100] = hunger * sensor_factor
            sensor_inj[100:120] = tired * sensor_factor

        # ---- brain step ----
        dopamine *= 0.985
        trace_pre *= TRACE_DECAY
        trace_post *= TRACE_DECAY
        energy += (1.0 - energy) * 0.001
        fatigue *= 0.98
        excitability = energy * (1.0 - fatigue * 0.5)

        syn_w = weight if dale else np.abs(weight)
        I_syn = np.zeros(N)
        np.add.at(I_syn, post, syn_w * smoothed[pre])
        I_eff = (I_syn + sensor_inj) * excitability

        I_noise = rng.standard_normal(N) * noise_scale
        kick = rng.random(N) < rare_p
        I_noise = np.where(kick, I_noise + 15.0, I_noise)
        I = I_eff + I_noise

        throttle = fatigue > 0.6
        if throttle.any():
            I[throttle] *= (1.0 - (fatigue[throttle] - 0.6) * 0.5)

        for _ in range(2):
            v += 0.5 * (0.04 * v * v + 5 * v + 140 - u + I)
            u += 0.5 * a * (b * v - u)
        np.clip(v, -90.0, 35.0, out=v)

        fired = v >= 30.0
        is_spike = fired.astype(np.float64)
        smoothed = smoothed * 0.8 + is_spike * 0.2
        firing_rate = firing_rate * 0.99 + is_spike * 0.01

        if fired.any():
            v[fired] = c[fired]
            u[fired] += d[fired]
            fatigue[fired] = np.minimum(1.0, fatigue[fired] + 0.05)
            energy[fired] = np.maximum(0.0, energy[fired] - 0.002)
            trace_pre[fired] += 1.0
            trace_post[fired] += 1.0

            if stdp_on:
                dopa_mult = 0.1 + dopamine * 4.0   # reward-modulated
                fired_idx = np.flatnonzero(fired)
                le = np.minimum(energy[pre], energy[post])  # local energy gate
                # potentiation: postsynaptic neuron fired
                pf = np.isin(post, fired_idx) & plastic
                if pf.any():
                    weight[pf] += A_PLUS * trace_pre[pre[pf]] * le[pf] * dopa_mult
                # depression: presynaptic neuron fired
                qf = np.isin(pre, fired_idx) & plastic
                if qf.any():
                    weight[qf] -= A_MINUS * trace_post[post[qf]] * le[qf] * dopa_mult
                weight[exc_syn] = np.clip(weight[exc_syn], 0.0, W_MAX)
                weight[inh_syn] = np.clip(weight[inh_syn], -W_MAX, 0.0)

        # homeostatic synaptic scaling every 500 steps (plastic synapses only)
        if stdp_on and (t + 1) % 500 == 0:
            rate_post = firing_rate[post]
            sf = np.ones(len(weight))
            sf[rate_post < 0.02] = 1.0 + 0.05 * HOMEO_STRENGTH
            sf[rate_post > 0.02] = max(0.7, 1.0 - 0.05 * HOMEO_STRENGTH)
            sf[~plastic] = 1.0
            weight *= sf
            weight[exc_syn] = np.clip(weight[exc_syn], 0.0, W_MAX)
            weight[inh_syn] = np.clip(weight[inh_syn], -W_MAX, 0.0)

        energy = np.clip(energy, 0.0, 1.0)
        fatigue = np.clip(fatigue, 0.0, 1.0)

        # ---- motor readout -> body ----
        if alive:
            mL = float(smoothed[mL0:mL1].mean())
            mR = float(smoothed[mR0:mR1].mean())
            fwd = (mL + mR) / 2.0
            speed = 0.0009 + 0.007 * fwd * MOTOR_GAIN
            if speed < 0:
                speed = 0.0
            to_center = np.arctan2(0.5 - cy, 0.5 - cx)
            dh_wall = _angwrap(to_center - head)
            turn = (mL - mR) * 0.9 * MOTOR_GAIN + dh_wall * W * 0.3
            head_vel = head_vel * 0.75 + turn * 0.25
            head += head_vel
            cx += np.cos(head) * speed
            cy += np.sin(head) * speed
            cx = min(0.97, max(0.03, cx)); cy = min(0.97, max(0.03, cy))
            distance += speed
            speed_sum += speed

            if len(food) > 0:
                dx = food[:, 0] - cx; dy = food[:, 1] - cy
                hit = np.flatnonzero(dx * dx + dy * dy < EAT_RADIUS2)
                if len(hit) > 0:
                    keep = np.ones(len(food), dtype=bool)
                    for h in hit:
                        keep[h] = False
                        score += 1
                        energy_val = min(100.0, energy_val + 28.0)
                        tiredness = max(0.0, tiredness - 0.04)
                        dopamine = min(2.0, dopamine + 0.85)
                    new = np.column_stack([rng.random(len(hit)) * 0.9 + 0.05,
                                           rng.random(len(hit)) * 0.9 + 0.05])
                    food = np.vstack([food[keep], new])

            energy_val -= 0.030 + 2.0 * speed + 400.0 * speed * speed
            tiredness = max(0.0, min(1.0, tiredness + speed * 3.0 - 0.008))
            survival_steps = t + 1
            if energy_val <= 0:
                energy_val = 0.0
                alive = False

        if t % record_every == 0:
            spikes[rec_idx] = fired
            weight_history[rec_idx] = np.mean(np.abs(weight))
            rec_idx += 1

    behavior = dict(
        food_eaten=int(score),
        distance_traveled=float(distance),
        mean_speed=float(speed_sum / max(1, survival_steps)),
        final_body_energy=float(energy_val),
        survival_steps=int(survival_steps),
        alive_end=bool(alive),
        food_per_1k_steps=float(score / (survival_steps / 1000.0)) if survival_steps > 0 else 0.0,
        food_per_distance=float(score / distance) if distance > 1e-9 else 0.0,
    )
    return dict(spikes=spikes[:rec_idx], weight_history=weight_history[:rec_idx],
                final_energy=energy, final_fatigue=fatigue, final_weight=weight,
                behavior=behavior)
