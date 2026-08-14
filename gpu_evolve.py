"""
GPU-batched evolution of the DigitalOrganoid creature brain.

Same biology as jules/organoid_simulator/creature_embodied.py (Izhikevich
RS/IB/FS/LTS neurons, Dale's principle, distance-based K-NN topology,
reward-modulated STDP, homeostatic scaling, the SYN_GAIN/MOTOR_SPEED_SCALE
conduction+motor fixes) but the WHOLE population x seed batch runs as one
set of vectorized tensor ops per timestep instead of a Python loop over
individuals. Topology (pre/post/is_reflex/types/a/b/c/d) is FIXED and shared
across the population -- only synaptic weight evolves -- so it batches
cleanly: one shared edge list, weight tensor shaped (batch, n_genes).

Deliberate simplifications vs. the CPU harness (same ones creature_embodied.py
already documents): no structural plasticity (fixed topology), no R-STDP
overlay (elig/signed/explore/shape) -- Task 5 found R-STDP added ~nothing on
top of evolution, so it's dropped here to keep the batched kernel simple.
Plain reward-modulated STDP + evolution + a reflex curriculum (new).

Curriculum: reflex_scale fades 1.0 -> 0.0 over CURRICULUM_GENS generations,
then holds at 0.0 for the remainder, so evolution/STDP have something to
shape around early instead of a blank no-reflex start on generation 0.

Usage:
    python gpu_evolve.py --pop 64 --gens 60 --device cuda
"""
import argparse
import math
import time
import numpy as np
import torch

# ---- constants, identical to creature_embodied.py -------------------------
W_MAX = 2.5
SYN_GAIN = 100.0
MOTOR_SPEED_SCALE = 0.10
TRACE_DECAY = 0.96
A_PLUS = 0.008
A_MINUS = 0.010
HOMEO_STRENGTH = 1.0
THETA_OFF = 0.6
SMELL_SCALE = 0.35
SENSOR_GAIN = 1.5
MOTOR_GAIN = 1.0
BASE_NOISE = 0.01
DEV_STAGE = 0.20
EAT_RADIUS2 = 0.0009
N_FOOD = 6
POOLS = {
    'sL': (0, 20), 'sR': (20, 40), 'sW': (40, 60),
    'sD': (60, 80), 'sH': (80, 100), 'sT': (100, 120),
    'mL': (120, 155), 'mR': (155, 190),
    # Fixed-retina "eye": N_EYE directionally-tuned photoreceptors spanning a
    # forward field of view, added alongside (not replacing) the existing
    # smell antennae. Carved from the interneuron budget so N stays 600.
    # This is preprocessing hardware, not perception -- each receptor gets a
    # raw intensity from its own preferred angle (same distance/cosine-lobe
    # physics the smell antennae already use, just resolved into many narrow
    # lobes instead of two wide ones), no food identity/segmentation, no
    # object detection. What that pattern MEANS is left entirely to the
    # downstream plastic network -- same emergence burden as the smell
    # sensors, just richer raw input to work with.
    'eye': (190, 190 + 20),
}
N_EYE = 20
EYE_FOV = 2.618  # ~150 degrees, forward-facing

SUBTYPE_RS, SUBTYPE_IB, SUBTYPE_FS, SUBTYPE_LTS = 0, 1, 2, 3
LONG_RANGE_P = 0.10
DECAY_SCALE = 0.6
TARGET_G = 1.4


# --------------------------------------------------------------------------- #
# Topology (built once on CPU with numpy, identical algorithm to
# creature_embodied.build_creature_network, then uploaded to the device).
# --------------------------------------------------------------------------- #
def build_scaffold_numpy(N, K, fi=0.2, seed=42, dense_k=0):
    rng = np.random.default_rng(seed)
    px = np.empty(N); py = np.empty(N); pz = np.empty(N)
    filled = 0
    while filled < N:
        cand = rng.random((N, 3)) * 2 - 1
        ok = (cand ** 2).sum(1) <= 1.0
        take = cand[ok]
        n = min(len(take), N - filled)
        px[filled:filled + n] = take[:n, 0]
        py[filled:filled + n] = take[:n, 1]
        pz[filled:filled + n] = take[:n, 2]
        filled += n

    types = np.where(rng.random(N) < fi, -1, 1).astype(np.int8)
    for (lo, hi) in POOLS.values():
        types[lo:hi] = 1

    a = np.zeros(N); b = np.zeros(N); c = np.zeros(N); d = np.zeros(N)
    for i in range(N):
        if types[i] > 0:
            base = (0.02, 0.2, -65, 8) if rng.random() < 0.85 else (0.02, 0.2, -50, 2)
        else:
            base = (0.1, 0.2, -65, 2) if rng.random() < 0.80 else (0.02, 0.25, -65, 2)
        jit = lambda: 1.0 + (rng.random() * 2 - 1) * 0.08
        a[i] = base[0] * jit(); b[i] = base[1] * jit()
        c[i] = base[2] + (rng.random() * 2 - 1) * 2.0
        d[i] = base[3] * jit()

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

    absum = np.zeros(N)
    np.add.at(absum, post, np.abs(weight))
    g = np.ones(N)
    nz = absum > 0
    g[nz] = TARGET_G / absum[nz]
    weight *= g[post]

    def link(sp, mp, wt, pre_l2, post_l2, w_l2, rfx):
        s0, s1 = POOLS[sp]; t0, t1 = POOLS[mp]
        for t in range(t0, t1):
            for s in range(s0, s1, 3):
                pre_l2.append(s); post_l2.append(t); w_l2.append(wt * types[s]); rfx.append(True)

    pre_l2 = []; post_l2 = []; w_l2 = []; rfx = []
    link('sL', 'mL', 0.6, pre_l2, post_l2, w_l2, rfx)
    link('sR', 'mR', 0.6, pre_l2, post_l2, w_l2, rfx)
    link('sW', 'mL', 0.4, pre_l2, post_l2, w_l2, rfx)
    link('sW', 'mR', 0.4, pre_l2, post_l2, w_l2, rfx)

    pre = np.concatenate([pre, np.array(pre_l2, dtype=np.int64)])
    post = np.concatenate([post, np.array(post_l2, dtype=np.int64)])
    weight = np.concatenate([weight, np.array(w_l2, dtype=np.float64)])
    is_reflex = np.concatenate([is_reflex, np.array(rfx, dtype=bool)])

    # Optional dense sensor->motor scaffold (round 2's condition): dense_k
    # random small-weight PLASTIC synapses per motor neuron from the sensor
    # pools, selection-shaped only (not hand-tuned toward chemotaxis), same
    # as breedstock_dense_noreflex.py. Kept separate from the reflex links
    # above -- these ARE plastic (is_reflex=False), evolution/STDP owns them.
    if dense_k > 0:
        sensor_pools = ['sL', 'sR', 'sW', 'sD', 'sH', 'sT']
        pre_d = []; post_d = []; w_d = []
        for mp in ('mL', 'mR'):
            t0, t1 = POOLS[mp]
            for tpos in range(t0, t1):
                for _ in range(dense_k):
                    sp = sensor_pools[rng.integers(len(sensor_pools))]
                    s0, s1 = POOLS[sp]
                    spos = rng.integers(s0, s1)
                    pre_d.append(spos); post_d.append(tpos)
                    w_d.append(abs(rng.standard_normal()) * types[spos] * 0.15)
        pre = np.concatenate([pre, np.array(pre_d, dtype=np.int64)])
        post = np.concatenate([post, np.array(post_d, dtype=np.int64)])
        weight = np.concatenate([weight, np.array(w_d, dtype=np.float64)])
        is_reflex = np.concatenate([is_reflex, np.zeros(len(pre_d), dtype=bool)])

    return dict(N=N, types=types, a=a, b=b, c=c, d=d,
                pre=pre, post=post, weight=weight, is_reflex=is_reflex)


# --------------------------------------------------------------------------- #
# Batched simulation. B = population x seeds. Topology shared; only the
# plastic-synapse weight tensor (B, n_plastic) differs per batch row.
# --------------------------------------------------------------------------- #
class BatchSim:
    def __init__(self, scaffold, device, dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        N = scaffold['N']
        self.N = N
        types = scaffold['types']
        pre = scaffold['pre']; post = scaffold['post']; is_reflex = scaffold['is_reflex']
        plastic = ~is_reflex

        t = lambda x, kind=dtype: torch.as_tensor(x, dtype=kind, device=device)
        self.a = t(scaffold['a']); self.b = t(scaffold['b'])
        self.c = t(scaffold['c']); self.d = t(scaffold['d'])

        self.pre_p = t(pre[plastic], torch.long)
        self.post_p = t(post[plastic], torch.long)
        self.pre_r = t(pre[is_reflex], torch.long)
        self.post_r = t(post[is_reflex], torch.long)
        self.reflex_base = t(scaffold['weight'][is_reflex])

        exc_p = types[pre][plastic] > 0
        self.exc_mask = t(exc_p, torch.bool)
        self.n_plastic = int(plastic.sum())

        (self.mL0, self.mL1) = POOLS['mL']
        (self.mR0, self.mR1) = POOLS['mR']
        (self.eye_lo, self.eye_hi) = POOLS['eye']
        self.theta_eye = torch.linspace(-EYE_FOV / 2, EYE_FOV / 2, N_EYE, device=device, dtype=dtype)
        self.pool_slices = {k: v for k, v in POOLS.items()}
        self.sensor_factor = SENSOR_GAIN * 14.0
        stage = DEV_STAGE
        self.noise_scale = BASE_NOISE + (1.0 - stage) * 0.015
        self.rare_p = 0.002 * (1.0 - stage)

    def run(self, weight_plastic, T, reflex_scale, stdp_on=True, record_steering=False,
            learning='stdp', elig_decay=0.995, elig_lr=0.03, value_lr=0.02,
            eat_radius2=None, n_food=None, food_spawn_radius=None, metab_scale=1.0,
            motor_noise_start=0.0, motor_noise_end=0.0, init_food=None, use_eye=False,
            fep_punish=False, fep_punish_t=150, fep_wall_thresh=0.7, fep_timeout_steps=800):
        """weight_plastic: (B, n_plastic) tensor, evolves in place and is
        returned updated. reflex_scale: python float, applied uniformly this
        lifetime (curriculum). Returns dict with food_eaten (B,) and, if
        record_steering, dL/dM traces for steering correlation.

        learning='stdp': original rule -- weight changes only at the instant
        pre/post fire, gated by *raw* dopamine level. Credits whichever
        synapse happened to be active in the same step as a reward event.

        learning='surprise': three-factor rule. `elig` is a slow trace
        (decay `elig_decay`, ~1/(1-elig_decay) step horizon) that accumulates
        Hebbian pre/post coincidences continuously, independent of reward --
        it remembers which synapses were active *on the way to* an eat event,
        not just at the instant of eating. A running value estimate `V`
        tracks expected reward (leaky average of dopamine); the actual weight
        update every step is elig * (dopamine - V) -- i.e. proportional to
        the SURPRISE (reward-prediction error), not the raw reward level, and
        applied via the eligibility trace so credit reaches back through the
        approach, not just the eat-instant.

        learning='fep': Free Energy Principle / predictability-based rule
        (DishBrain/bl1-inspired). No reward-gated weight update at all --
        plain, always-on Hebbian STDP (unmodulated by dopamine). Credit
        assignment instead happens entirely on the ENVIRONMENT side via
        fep_punish: when the creature hits a wall or goes too long without
        eating, informative sensor channels (smell L/R, eye) are replaced
        with pure random noise for fep_punish_t steps -- scrambling the
        signal Hebbian plasticity has to work with. Coherent, predictable
        input naturally gets reinforced by ordinary Hebbian correlation;
        scrambled input naturally doesn't. Orthogonal to `learning` in
        principle but intended to pair with learning='fep'.

        "Hunting bootcamp" environment knobs (all optional, default = normal
        difficulty): eat_radius2/n_food override the normal reward geometry;
        food_spawn_radius, if set, respawns food within that radius of the
        creature's CURRENT position instead of uniformly over the arena, so
        reward events happen constantly instead of rarely -- more grist for
        the credit-assignment machinery per unit of simulated time.
        metab_scale scales the energy cost (lets a naive/noisy creature
        survive long enough to learn something before starving).
        motor_noise_start/end: exploration noise added to the motor readout,
        linearly annealed across this call's T steps -- lots of undirected
        wandering-into-food early, tapering off so later steps reflect the
        network's own (increasingly learned) choices. init_food, if given, is
        used as the T=0 food layout instead of a fresh random one (for
        chaining a second run() call from where the first left off)."""
        dev, dt = self.device, self.dtype
        B = weight_plastic.shape[0]
        N = self.N
        wp = weight_plastic
        surprise = (learning == 'surprise')
        fep = (learning == 'fep')
        eat_r2 = EAT_RADIUS2 if eat_radius2 is None else eat_radius2
        nf = N_FOOD if n_food is None else n_food
        if surprise:
            elig = torch.zeros(B, self.n_plastic, device=dev, dtype=dt)
            value = torch.zeros(B, device=dev, dtype=dt)
        if fep_punish:
            steps_since_food = torch.zeros(B, device=dev, dtype=dt)
            punish_timer = torch.zeros(B, device=dev, dtype=dt)

        v = self.c.unsqueeze(0).expand(B, N).clone()
        u = (self.b * self.c).unsqueeze(0).expand(B, N).clone()
        energy = torch.ones(B, N, device=dev, dtype=dt)
        fatigue = torch.zeros(B, N, device=dev, dtype=dt)
        trace_pre = torch.zeros(B, N, device=dev, dtype=dt)
        trace_post = torch.zeros(B, N, device=dev, dtype=dt)
        smoothed = torch.zeros(B, N, device=dev, dtype=dt)
        firing_rate = torch.zeros(B, N, device=dev, dtype=dt)

        cx = torch.full((B,), 0.5, device=dev, dtype=dt)
        cy = torch.full((B,), 0.5, device=dev, dtype=dt)
        head = torch.rand(B, device=dev, dtype=dt) * 6.28
        head_vel = torch.zeros(B, device=dev, dtype=dt)
        energy_val = torch.full((B,), 100.0, device=dev, dtype=dt)
        tiredness = torch.zeros(B, device=dev, dtype=dt)
        dopamine = torch.zeros(B, device=dev, dtype=dt)
        alive = torch.ones(B, device=dev, dtype=torch.bool)
        score = torch.zeros(B, device=dev, dtype=dt)
        survival_steps = torch.zeros(B, device=dev, dtype=dt)

        if init_food is not None:
            food = init_food.clone()
        else:
            food = torch.rand(B, nf, 2, device=dev, dtype=dt) * 0.9 + 0.05

        dL_hist = [] if record_steering else None
        dM_hist = [] if record_steering else None
        dEye_hist = [] if (record_steering and use_eye) else None

        for t in range(T):
            dx = food[:, :, 0] - cx.unsqueeze(1)
            dy = food[:, :, 1] - cy.unsqueeze(1)
            dist = torch.sqrt(dx * dx + dy * dy)
            nearest = dist.min(dim=1).values
            r = dist / SMELL_SCALE
            intensity = 1.0 / (1.0 + r * r)
            ang = torch.atan2(torch.sin(torch.atan2(dy, dx) - head.unsqueeze(1)),
                              torch.cos(torch.atan2(dy, dx) - head.unsqueeze(1)))
            L = (intensity * torch.clamp(torch.cos(ang - THETA_OFF), min=0.0)).sum(dim=1)
            R = (intensity * torch.clamp(torch.cos(ang + THETA_OFF), min=0.0)).sum(dim=1)
            L = 1.0 - torch.exp(-L); R = 1.0 - torch.exp(-R)
            if use_eye:
                # Same physics as L/R (distance falloff x cosine directional
                # lobe), just N_EYE narrow lobes across a wide FOV instead of
                # two wide ones -- a fixed retina, not a detector.
                eye_ang = ang.unsqueeze(-1) - self.theta_eye.view(1, 1, -1)  # (B,NF,N_EYE)
                eye_field = (intensity.unsqueeze(-1) * torch.clamp(torch.cos(eye_ang), min=0.0)).sum(dim=1)
                eye_sig = 1.0 - torch.exp(-eye_field)
                eye_sig = torch.clamp(eye_sig + (torch.rand(B, N_EYE, device=dev, dtype=dt) - 0.5) * 0.04, 0.0, 1.0)
            L = torch.clamp(L + (torch.rand(B, device=dev, dtype=dt) - 0.5) * 0.04, 0.0, 1.0)
            R = torch.clamp(R + (torch.rand(B, device=dev, dtype=dt) - 0.5) * 0.04, 0.0, 1.0)
            wall = torch.clamp(1.0 - torch.minimum(torch.minimum(cx, cy),
                               torch.minimum(1.0 - cx, 1.0 - cy)) / 0.15, min=0.0)
            food_close = 1.0 / (1.0 + (nearest / 0.3) ** 2)
            hunger = 1.0 - energy_val / 100.0
            tired = torch.clamp(tiredness, max=1.0)

            if fep_punish:
                trigger = (wall > fep_wall_thresh) | (steps_since_food > fep_timeout_steps)
                punish_timer = torch.where(trigger, torch.full_like(punish_timer, float(fep_punish_t)), punish_timer)
                punished = punish_timer > 0
                L_inj = torch.where(punished, torch.rand(B, device=dev, dtype=dt), L)
                R_inj = torch.where(punished, torch.rand(B, device=dev, dtype=dt), R)
                if use_eye:
                    eye_inj = torch.where(punished.unsqueeze(1), torch.rand(B, N_EYE, device=dev, dtype=dt), eye_sig)
            else:
                L_inj, R_inj = L, R
                if use_eye:
                    eye_inj = eye_sig

            sensor_inj = torch.zeros(B, N, device=dev, dtype=dt)
            av = alive.to(dt).unsqueeze(1)
            sensor_inj[:, 0:20] = (L_inj * self.sensor_factor).unsqueeze(1) * av
            sensor_inj[:, 20:40] = (R_inj * self.sensor_factor).unsqueeze(1) * av
            sensor_inj[:, 40:60] = (wall * self.sensor_factor).unsqueeze(1) * av
            sensor_inj[:, 60:80] = (food_close * self.sensor_factor).unsqueeze(1) * av
            if use_eye:
                sensor_inj[:, self.eye_lo:self.eye_hi] = eye_inj * self.sensor_factor * av
            sensor_inj[:, 80:100] = (hunger * self.sensor_factor).unsqueeze(1) * av
            sensor_inj[:, 100:120] = (tired * self.sensor_factor).unsqueeze(1) * av

            dopamine = dopamine * 0.985
            trace_pre = trace_pre * TRACE_DECAY
            trace_post = trace_post * TRACE_DECAY
            energy = energy + (1.0 - energy) * 0.001
            fatigue = fatigue * 0.98
            excitability = energy * (1.0 - fatigue * 0.5)

            I_syn = torch.zeros(B, N, device=dev, dtype=dt)
            I_syn.index_add_(1, self.post_p, wp * smoothed[:, self.pre_p])
            reflex_w = self.reflex_base * reflex_scale
            I_syn.index_add_(1, self.post_r, reflex_w.unsqueeze(0) * smoothed[:, self.pre_r])
            I_eff = (I_syn * SYN_GAIN + sensor_inj) * excitability

            I_noise = torch.randn(B, N, device=dev, dtype=dt) * self.noise_scale
            kick = torch.rand(B, N, device=dev, dtype=dt) < self.rare_p
            I_noise = torch.where(kick, I_noise + 15.0, I_noise)
            I = I_eff + I_noise
            throttle = fatigue > 0.6
            I = torch.where(throttle, I * (1.0 - (fatigue - 0.6) * 0.5), I)

            for _ in range(2):
                v = v + 0.5 * (0.04 * v * v + 5 * v + 140 - u + I)
                u = u + 0.5 * self.a.unsqueeze(0) * (self.b.unsqueeze(0) * v - u)
            v = torch.clamp(v, -90.0, 35.0)

            fired = v >= 30.0
            firef = fired.to(dt)
            smoothed = smoothed * 0.8 + firef * 0.2
            firing_rate = firing_rate * 0.99 + firef * 0.01

            v = torch.where(fired, self.c.unsqueeze(0).expand(B, N), v)
            u = torch.where(fired, u + self.d.unsqueeze(0), u)
            fatigue = torch.where(fired, torch.clamp(fatigue + 0.05, max=1.0), fatigue)
            energy = torch.where(fired, torch.clamp(energy - 0.002, min=0.0), energy)
            trace_pre = trace_pre + firef
            trace_post = trace_post + firef

            if stdp_on and not surprise and not fep:
                dopa_mult = (0.1 + dopamine * 4.0).unsqueeze(1)
                pf = fired[:, self.post_p]
                qf = fired[:, self.pre_p]
                le = torch.minimum(energy[:, self.pre_p], energy[:, self.post_p])
                wp = wp + A_PLUS * trace_pre[:, self.pre_p] * le * dopa_mult * pf.to(dt)
                wp = wp - A_MINUS * trace_post[:, self.post_p] * le * dopa_mult * qf.to(dt)
                wp = torch.where(self.exc_mask.unsqueeze(0),
                                 torch.clamp(wp, 0.0, W_MAX),
                                 torch.clamp(wp, -W_MAX, 0.0))

            if stdp_on and fep:
                # Plain, unmodulated Hebbian STDP -- no dopamine gating at
                # all. All credit assignment happens on the environment side
                # (fep_punish scrambling), not in the weight-update rule.
                pf = fired[:, self.post_p]
                qf = fired[:, self.pre_p]
                le = torch.minimum(energy[:, self.pre_p], energy[:, self.post_p])
                wp = wp + A_PLUS * trace_pre[:, self.pre_p] * le * pf.to(dt)
                wp = wp - A_MINUS * trace_post[:, self.post_p] * le * qf.to(dt)
                wp = torch.where(self.exc_mask.unsqueeze(0),
                                 torch.clamp(wp, 0.0, W_MAX),
                                 torch.clamp(wp, -W_MAX, 0.0))

            if stdp_on and surprise:
                # 1) Hebbian eligibility: slow trace of pre/post coincidence,
                #    NOT gated by reward -- keeps "who was active on approach"
                #    alive for ~1/(1-elig_decay) steps after the fact.
                pf = fired[:, self.post_p]
                qf = fired[:, self.pre_p]
                le = torch.minimum(energy[:, self.pre_p], energy[:, self.post_p])
                hebb = (A_PLUS * trace_pre[:, self.pre_p] * pf.to(dt)
                       - A_MINUS * trace_post[:, self.post_p] * qf.to(dt)) * le
                elig = elig * elig_decay + hebb
                # 2) Value prediction + surprise: V tracks expected dopamine;
                #    delta is the reward-prediction error (Schultz-style).
                delta = dopamine - value
                value = value + value_lr * delta
                # 3) Weight update = eligibility x surprise, every step -- so
                #    credit assignment reaches back into the trace, not just
                #    synapses active in the exact instant reward landed.
                wp = wp + elig_lr * elig * delta.unsqueeze(1)
                wp = torch.where(self.exc_mask.unsqueeze(0),
                                 torch.clamp(wp, 0.0, W_MAX),
                                 torch.clamp(wp, -W_MAX, 0.0))

            if stdp_on:
                if (t + 1) % 500 == 0:
                    rate_post = firing_rate[:, self.post_p]
                    sf = torch.ones_like(rate_post)
                    sf = torch.where(rate_post < 0.02, torch.full_like(sf, 1.0 + 0.05 * HOMEO_STRENGTH), sf)
                    sf = torch.where(rate_post > 0.02, torch.full_like(sf, max(0.7, 1.0 - 0.05 * HOMEO_STRENGTH)), sf)
                    wp = wp * sf
                    wp = torch.where(self.exc_mask.unsqueeze(0),
                                     torch.clamp(wp, 0.0, W_MAX),
                                     torch.clamp(wp, -W_MAX, 0.0))

            energy = torch.clamp(energy, 0.0, 1.0)
            fatigue = torch.clamp(fatigue, 0.0, 1.0)

            mL = smoothed[:, self.mL0:self.mL1].mean(dim=1)
            mR = smoothed[:, self.mR0:self.mR1].mean(dim=1)
            if motor_noise_start or motor_noise_end:
                frac = t / max(1, T - 1)
                noise_t = motor_noise_start + (motor_noise_end - motor_noise_start) * frac
                mL = mL + torch.randn(B, device=dev, dtype=dt) * noise_t
                mR = mR + torch.randn(B, device=dev, dtype=dt) * noise_t
            if record_steering:
                dL_hist.append((L - R).detach())
                dM_hist.append((mL - mR).detach())
                if use_eye:
                    half = N_EYE // 2
                    dEye_hist.append((eye_sig[:, :half].mean(dim=1) - eye_sig[:, half:].mean(dim=1)).detach())
            fwd = (mL + mR) / 2.0
            speed = torch.clamp(0.0009 + MOTOR_SPEED_SCALE * fwd * MOTOR_GAIN, min=0.0)
            to_center = torch.atan2(0.5 - cy, 0.5 - cx)
            dh_wall = torch.atan2(torch.sin(to_center - head), torch.cos(to_center - head))
            turn = (mL - mR) * 0.9 * MOTOR_GAIN + dh_wall * wall * 0.3
            head_vel = head_vel * 0.75 + turn * 0.25
            new_head = head + head_vel
            new_cx = torch.clamp(cx + torch.cos(new_head) * speed, 0.03, 0.97)
            new_cy = torch.clamp(cy + torch.sin(new_head) * speed, 0.03, 0.97)

            av_f = alive.to(dt)
            head = torch.where(alive, new_head, head)
            cx = torch.where(alive, new_cx, cx)
            cy = torch.where(alive, new_cy, cy)
            head_vel = head_vel * av_f

            dxh = food[:, :, 0] - cx.unsqueeze(1)
            dyh = food[:, :, 1] - cy.unsqueeze(1)
            hit = (dxh * dxh + dyh * dyh < eat_r2) & alive.unsqueeze(1)
            hits = hit.sum(dim=1).to(dt)
            if food_spawn_radius is not None:
                offset = (torch.rand(B, nf, 2, device=dev, dtype=dt) * 2 - 1) * food_spawn_radius
                center = torch.stack([cx, cy], dim=1).unsqueeze(1)
                new_food = torch.clamp(center + offset, 0.03, 0.97)
            else:
                new_food = torch.rand(B, nf, 2, device=dev, dtype=dt) * 0.9 + 0.05
            food = torch.where(hit.unsqueeze(-1), new_food, food)
            score = score + hits
            energy_val = torch.clamp(energy_val + 28.0 * hits, max=100.0)
            tiredness = torch.clamp(tiredness - 0.04 * hits, min=0.0)
            dopamine = torch.clamp(dopamine + 0.85 * hits, max=2.0)

            energy_val = torch.where(alive, energy_val - metab_scale * (0.030 + 2.0 * speed + 400.0 * speed * speed), energy_val)
            tiredness = torch.where(alive, torch.clamp(tiredness + speed * 3.0 - 0.008, 0.0, 1.0), tiredness)
            survival_steps = survival_steps + av_f
            alive = alive & (energy_val > 0)

            if fep_punish:
                steps_since_food = torch.where(hit.any(dim=1), torch.zeros_like(steps_since_food), steps_since_food + 1.0)
                punish_timer = torch.clamp(punish_timer - 1.0, min=0.0)

        out = dict(food_eaten=score, weight_plastic=wp, survival_steps=survival_steps)
        if record_steering:
            out['dL'] = torch.stack(dL_hist, dim=0)   # (T, B)
            out['dM'] = torch.stack(dM_hist, dim=0)
            if use_eye:
                out['dEye'] = torch.stack(dEye_hist, dim=0)
        return out


# --------------------------------------------------------------------------- #
# GA operators (numpy, CPU side -- cheap relative to the GPU sim).
# --------------------------------------------------------------------------- #
def clamp_genome(genome, lo, hi):
    return np.clip(genome, lo, hi)


def crossover(p1, p2, lo, hi, rng):
    mask = rng.random(len(p1)) < 0.5
    child = np.where(mask, p1, p2)
    return clamp_genome(child, lo, hi)


def mutate(genome, lo, hi, sigma, rate, rng):
    child = genome.copy()
    hit = rng.random(len(genome)) < rate
    child[hit] += rng.standard_normal(int(hit.sum())) * sigma
    return clamp_genome(child, lo, hi)


def breed_next_generation(ranked_pop, lo, hi, pop_size, elite_frac, sigma, mut_rate, crossover_p, rng):
    n_elite = max(2, int(round(pop_size * elite_frac)))
    parents = ranked_pop[:n_elite]
    next_pop = [parents[0].copy()]
    while len(next_pop) < pop_size:
        if rng.random() < crossover_p and len(parents) >= 2:
            i, j = rng.choice(len(parents), size=2, replace=False)
            child = crossover(parents[i], parents[j], lo, hi, rng)
        else:
            child = parents[rng.integers(len(parents))].copy()
        child = mutate(child, lo, hi, sigma, mut_rate, rng)
        next_pop.append(child)
    return next_pop


def steering_correlation(dL, dM):
    """dL, dM: (T, B) tensors. Returns per-batch Pearson r, (B,) numpy."""
    dL = dL - dL.mean(dim=0, keepdim=True)
    dM = dM - dM.mean(dim=0, keepdim=True)
    num = (dL * dM).sum(dim=0)
    den = torch.sqrt((dL * dL).sum(dim=0) * (dM * dM).sum(dim=0)) + 1e-9
    return (num / den).cpu().numpy()


def oracle(T, seed, speed=0.0009):
    rng = np.random.default_rng(seed)
    cx, cy = 0.5, 0.5
    E, score = 100.0, 0
    food = np.column_stack([rng.random(6) * 0.9 + 0.05, rng.random(6) * 0.9 + 0.05])
    for t in range(T):
        dx, dy = food[:, 0] - cx, food[:, 1] - cy
        i = (dx * dx + dy * dy).argmin()
        head = np.arctan2(dy[i], dx[i])
        cx = min(0.97, max(0.03, cx + np.cos(head) * speed))
        cy = min(0.97, max(0.03, cy + np.sin(head) * speed))
        hit = np.flatnonzero((food[:, 0] - cx) ** 2 + (food[:, 1] - cy) ** 2 < 0.0009)
        if len(hit):
            keep = np.ones(len(food), bool)
            for h in hit:
                keep[h] = False; score += 1; E = min(100.0, E + 28.0)
            food = np.vstack([food[keep], np.column_stack([rng.random(len(hit)) * 0.9 + 0.05,
                                                            rng.random(len(hit)) * 0.9 + 0.05])])
        E -= 0.030 + 2.0 * speed + 400 * speed * speed
        if E <= 0:
            return score
    return score


def run_lifetime(sim, weight_plastic, args, reflex_scale, T, record_steering=False):
    """One genome-batch's full lifetime: optional bootcamp phase (easy,
    dense-reward, annealed exploration -- shapes weights) followed by the
    real (scored) phase at normal difficulty. Only the real phase's
    food_eaten/steering counts for fitness -- bootcamp food doesn't."""
    wp = weight_plastic
    fep_kw = dict(fep_punish=args.fep_punish, fep_punish_t=args.fep_punish_t,
                  fep_wall_thresh=args.fep_wall_thresh, fep_timeout_steps=args.fep_timeout_steps)
    if args.bootcamp:
        boot_out = sim.run(wp, args.boot_t, reflex_scale, stdp_on=True,
                           learning=args.learning, elig_decay=args.elig_decay,
                           elig_lr=args.elig_lr, value_lr=args.value_lr,
                           eat_radius2=args.boot_eat_r2, n_food=args.boot_n_food,
                           food_spawn_radius=args.boot_food_radius,
                           metab_scale=args.boot_metab_scale,
                           motor_noise_start=args.boot_noise_start,
                           motor_noise_end=args.boot_noise_end,
                           use_eye=args.use_eye, **fep_kw)
        wp = boot_out['weight_plastic']
    return sim.run(wp, T, reflex_scale, stdp_on=True, record_steering=record_steering,
                   learning=args.learning, elig_decay=args.elig_decay,
                   elig_lr=args.elig_lr, value_lr=args.value_lr, use_eye=args.use_eye, **fep_kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pop', type=int, default=64)
    ap.add_argument('--gens', type=int, default=60)
    ap.add_argument('--curriculum-gens', type=int, default=40,
                    help='generations over which reflex_scale fades 1.0->0.0')
    ap.add_argument('--train-seeds', type=int, default=4)
    ap.add_argument('--train-t', type=int, default=3000)
    ap.add_argument('--heldout-seeds', type=int, default=8)
    ap.add_argument('--heldout-t', type=int, default=6000)
    ap.add_argument('--elite-frac', type=float, default=0.25)
    ap.add_argument('--mut-sigma', type=float, default=0.08)
    ap.add_argument('--mut-rate', type=float, default=0.5)
    ap.add_argument('--crossover-p', type=float, default=0.9)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--N', type=int, default=600)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--scaffold-seed', type=int, default=42)
    ap.add_argument('--out', type=str, default='champion_gpu_curriculum.npy')
    ap.add_argument('--dense-k', type=int, default=0,
                    help='extra plastic sensor->motor synapses per motor neuron (round 2 scaffold)')
    ap.add_argument('--learning', type=str, default='stdp', choices=['stdp', 'surprise', 'fep'],
                    help='stdp = reward-gated instantaneous STDP; surprise = eligibility x TD-error; '
                         'fep = plain unmodulated Hebbian STDP, pair with --fep-punish')
    ap.add_argument('--elig-decay', type=float, default=0.995)
    ap.add_argument('--elig-lr', type=float, default=0.03)
    ap.add_argument('--value-lr', type=float, default=0.02)
    ap.add_argument('--fixed-reflex-scale', type=float, default=None,
                    help='if set, overrides curriculum and holds reflex_scale constant all generations')
    ap.add_argument('--bootcamp', action='store_true',
                    help='easy-mode pretraining phase before the real (scored) lifetime: '
                         'dense nearby food, bigger eat radius, low metabolism, annealed '
                         'exploration noise -- shapes weights with lots of reward signal '
                         'before the sparse real environment')
    ap.add_argument('--boot-t', type=int, default=1500)
    ap.add_argument('--boot-eat-r2', type=float, default=0.006, help='~2.6x the real eat radius')
    ap.add_argument('--boot-n-food', type=int, default=6)
    ap.add_argument('--boot-food-radius', type=float, default=0.15,
                    help='food always respawns within this radius of the creature')
    ap.add_argument('--boot-metab-scale', type=float, default=0.4)
    ap.add_argument('--boot-noise-start', type=float, default=0.35)
    ap.add_argument('--boot-noise-end', type=float, default=0.05)
    ap.add_argument('--use-eye', action='store_true',
                    help='add the fixed-retina eye pool (N_EYE directional photoreceptors) '
                         'alongside the existing smell antennae')
    ap.add_argument('--fep-punish', action='store_true',
                    help='FEP-style environment punishment: wall hits / going too long without '
                         'food trigger a burst of pure noise on smell L/R (and eye, if used), '
                         'replacing the informative signal -- pairs with --learning fep')
    ap.add_argument('--fep-punish-t', type=int, default=150, help='punishment burst duration, steps')
    ap.add_argument('--fep-wall-thresh', type=float, default=0.7,
                    help='wall-proximity signal (0-1) that triggers punishment')
    ap.add_argument('--fep-timeout-steps', type=int, default=800,
                    help='steps without eating that trigger punishment')
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f'device: {device}', flush=True)
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    print(f'learning: {args.learning}  dense_k: {args.dense_k}', flush=True)

    scaffold = build_scaffold_numpy(args.N, args.K, fi=0.2, seed=args.scaffold_seed, dense_k=args.dense_k)
    sim = BatchSim(scaffold, device)
    n_genes = sim.n_plastic
    print(f'genes (plastic synapses): {n_genes}', flush=True)

    plastic = ~scaffold['is_reflex']
    types = scaffold['types']; pre = scaffold['pre']
    exc = types[pre][plastic] > 0
    lo = np.where(exc, 0.0, -W_MAX)
    hi = np.where(exc, W_MAX, 0.0)
    base_genome = scaffold['weight'][plastic].copy()

    rng = np.random.default_rng(0)
    pop = [base_genome.copy()]
    for _ in range(args.pop - 1):
        g = base_genome + rng.standard_normal(n_genes) * args.mut_sigma
        pop.append(clamp_genome(g, lo, hi))

    ceiling = np.mean([oracle(6000, s) for s in range(1, 7)])
    print(f'oracle ceiling (6000 steps): {ceiling:.1f} food', flush=True)

    t0 = time.time()
    champion, champion_fit = None, -1e9
    S = args.train_seeds
    for gen in range(args.gens):
        if args.fixed_reflex_scale is not None:
            reflex_scale = args.fixed_reflex_scale
        else:
            reflex_scale = max(0.0, 1.0 - gen / max(1, args.curriculum_gens))

        pop_t = torch.tensor(np.stack(pop), dtype=torch.float32, device=device)  # (P, E)
        P = pop_t.shape[0]
        batch_w = pop_t.repeat_interleave(S, dim=0)  # (P*S, E)

        out = run_lifetime(sim, batch_w, args, reflex_scale, args.train_t)
        food = out['food_eaten'].view(P, S).mean(dim=1).cpu().numpy()

        order = np.argsort(-food)
        ranked = [pop[i] for i in order]
        scored_food = food[order]
        best_fit = float(scored_food[0])
        mean_fit = float(food.mean())
        if best_fit > champion_fit:
            champion_fit = best_fit
            champion = ranked[0].copy()

        print(f'gen {gen:3d}  reflex_scale {reflex_scale:.2f}  best {best_fit:6.2f}  '
              f'mean {mean_fit:6.2f}  champion-so-far {champion_fit:6.2f}  '
              f't={time.time()-t0:6.1f}s', flush=True)

        pop = breed_next_generation(ranked, lo, hi, args.pop, args.elite_frac,
                                    args.mut_sigma, args.mut_rate, args.crossover_p, rng)

    np.save(args.out, champion)
    print(f'\nchampion saved to {args.out}', flush=True)

    # Held-out evaluation at reflex_scale=0.0 (the real no-reflex test) plus a
    # steering-correlation check on the same held-out champion.
    champ_t = torch.tensor(champion, dtype=torch.float32, device=device).unsqueeze(0)
    champ_batch = champ_t.repeat(args.heldout_seeds, 1)
    out = run_lifetime(sim, champ_batch, args, reflex_scale=0.0, T=args.heldout_t, record_steering=True)
    foods = out['food_eaten'].cpu().numpy()
    r = steering_correlation(out['dL'], out['dM'])
    print(f'\nFINAL held-out ({args.heldout_seeds} seeds, T={args.heldout_t}, reflex=0): '
          f'food {foods.mean():.2f} +/- {foods.std()/math.sqrt(len(foods)):.2f}')
    print(f'per-seed food: {list(foods)}')
    print(f'steering correlation (smell), per-seed: {list(np.round(r, 3))}')
    print(f'steering correlation (smell), mean: {r.mean():.3f}')
    if args.use_eye:
        r_eye = steering_correlation(out['dEye'], out['dM'])
        print(f'steering correlation (EYE), per-seed: {list(np.round(r_eye, 3))}')
        print(f'steering correlation (EYE), mean: {r_eye.mean():.3f}')
    print(f'oracle ceiling: {ceiling:.1f}')


if __name__ == '__main__':
    main()
