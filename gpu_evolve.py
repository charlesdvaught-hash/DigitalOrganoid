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

Task 13 (CANDIDATE_MECHANISMS.md #0, #2): added an asym weight-space
diagnostic (mean(w[sL->mL]) - mean(w[sL->mR]), and the sR mirror) logged
per generation and on the final held-out champion, plus a --homeo {mult,sub}
flag. 'mult' is the original multiplicative, 500-step homeostasis (default,
reproduces every prior task's results unchanged). 'sub' is Miller & MacKay
(1994)-style SUBTRACTIVE normalization of each postsynaptic neuron's total
incoming excitatory weight, applied every --homeo-every steps (default 20,
~25x faster than 'mult''s 500) -- the form that can break L/R symmetry,
unlike multiplicative scaling which provably preserves it.

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

        # ---- Task 13: cached bits for --homeo sub, computed once (shared
        # topology across the population) -- which plastic-excitatory
        # synapses feed each postsynaptic neuron, and how many. ----
        exc_p_idx = np.flatnonzero(exc_p)
        self.post_exc = self.post_p[exc_p_idx] if len(exc_p_idx) else torch.empty(0, dtype=torch.long, device=device)
        # exc_p_idx indexes into the (n_plastic,) axis directly (same order
        # exc_mask does), so wp[:, self.exc_mask] and self.post_exc line up.
        fan_in_exc = torch.zeros(N, device=device, dtype=dtype)
        if len(exc_p_idx):
            fan_in_exc.index_add_(0, self.post_exc, torch.ones_like(self.post_exc, dtype=dtype))
        self.fan_in_exc = torch.clamp(fan_in_exc, min=1.0)

        # ---- Task 13 candidate #4 (evolve-the-plasticity-rule): per-synapse
        # pool-pair index (for gathering evolved per-pair Hebbian rates) and
        # the topology's own fixed initial weight (used as the shared,
        # non-evolved starting point for --learning evolverule). ----
        pair_idx, n_pairs, pair_labels = build_pair_map(scaffold)
        self.pair_idx = t(pair_idx, torch.long)
        self.n_pairs = n_pairs
        self.pair_labels = pair_labels
        self.init_weight = t(scaffold['weight'][plastic])

        # ---- Task 13 candidate A (evo-devo guidance-code wiring) ----
        pool_id = build_pool_id(N)
        pre_pool_id_np = pool_id[pre[plastic]]
        post_pool_id_np = pool_id[post[plastic]]
        self.pre_pool_id = t(pre_pool_id_np, torch.long)
        self.post_pool_id = t(post_pool_id_np, torch.long)
        self.n_pools = N_POOLS

        # ---- Task 17 (Dave's "solid starting structure" -- contralateral
        # wiring bias): which plastic synapses are one of the 4 sensor<->
        # motor pool pairs (sL-mL, sL-mR, sR-mL, sR-mR)? Gives evolution a
        # direct, dedicated, low-dimensional knob for exactly the structural
        # degree of freedom real nervous systems specialize (decussation),
        # instead of hoping the ambient per-pool code-compatibility formula
        # stumbles onto it. lat_idx is -1 (no bias applies) for every other
        # synapse; lat_mask marks which ones DO apply. ----
        lat_idx_np = build_lat_idx(pre_pool_id_np, post_pool_id_np)
        self.lat_idx = t(lat_idx_np, torch.long)
        self.lat_mask = t(lat_idx_np >= 0, torch.bool)

    def gate_weights(self, base_weight, guide_code, gain=4.0, lat_bias=None):
        """base_weight: (n_plastic,) fixed sign+magnitude init (e.g.
        self.init_weight). guide_code: (B, n_pools, code_dim), evolved.
        Returns (B, n_plastic) gated initial weight: base_weight * sigmoid(
        gain * dot(code[pre_pool], code[post_pool])) -- pools whose evolved
        codes are compatible (high dot product) start strongly connected;
        incompatible ones start near zero. gain=0 / code all-zero collapses
        to gate=0.5 everywhere (symmetric, no bias -- the neutral start).

        lat_bias: optional (B, 4) evolved genes, one per sensor<->motor pool
        pair (sL-mL, sL-mR, sR-mL, sR-mR order, see LAT_PAIRS) -- an
        additional additive logit shift applied ONLY to synapses in one of
        those 4 pairs, on top of the ordinary code-compatibility gate.
        Structural (weight-space initial-strength bias, evolved not
        hand-set), not a computed control law -- lifetime plasticity is
        still free to overwrite it entirely. Default None leaves every
        prior result unchanged (verified no-op)."""
        code_pre = guide_code[:, self.pre_pool_id, :]    # (B, n_plastic, code_dim)
        code_post = guide_code[:, self.post_pool_id, :]
        compat = (code_pre * code_post).sum(dim=-1)       # (B, n_plastic)
        logit = gain * compat
        if lat_bias is not None:
            lat_idx_safe = self.lat_idx.clamp(min=0)
            lat_term = lat_bias[:, lat_idx_safe] * self.lat_mask.unsqueeze(0).to(lat_bias.dtype)
            logit = logit + lat_term
        gate = torch.sigmoid(logit)
        return base_weight.unsqueeze(0) * gate

    def run(self, weight_plastic, T, reflex_scale, stdp_on=True, record_steering=False,
            learning='stdp', elig_decay=0.995, elig_lr=0.03, value_lr=0.02,
            eat_radius2=None, n_food=None, food_spawn_radius=None, metab_scale=1.0,
            motor_noise_start=0.0, motor_noise_end=0.0, init_food=None, use_eye=False,
            fep_punish=False, fep_punish_t=150, fep_wall_thresh=0.7, fep_timeout_steps=800,
            homeo='mult', homeo_every=20, eh_lr=0.05, eh_bar_decay=0.8,
            cp_steps=10**9, cp_decay_len=1, cp_floor=1.0, fep_punish_gate_steps=0,
            btsp_decay=0.999, btsp_lr=0.05, rule_a_plus=None, rule_a_minus=None,
            clamp_LR=None, freeze_motion=False):
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

        learning='eh': Exploratory Hebbian / reward-modulated Hebbian with
        double baseline subtraction (Legenstein et al. 2010; Hoerzer et al.
        2014, CANDIDATE_MECHANISMS.md #1). Delta_w = eh_lr * (dopamine -
        dopa_bar) * x_j * (a_i - a_bar_i) -- unlike 'stdp'/'surprise'/'fep',
        the postsynaptic term is a SIGNED fluctuation from that neuron's own
        recent running average, not a coincidence count, so it can push mL
        up and mR down on the same input/step.

        learning='btsp': behavioural-timescale eligibility (CANDIDATE_
        MECHANISMS.md #B2, Bittner et al. 2017-inspired). A very slow
        (btsp_decay, default 0.999 -> ~1000-step horizon, matching this
        sim's actual approach-and-eat behavioural loop length) Hebbian
        coincidence trace accumulates every step, independent of reward --
        same math as 'surprise's `elig` but a 5x longer horizon. Unlike
        'surprise', the trace is applied to weights ONLY on an eat-event
        step (Delta_w = btsp_lr * elig, then the trace resets), not every
        step -- so it never fires during a fep_punish window (those trigger
        on wall/timeout, not on eating), sidestepping the interference that
        sank Task 10's continuous 'surprise' + fep-punish combo.

        cp_steps/cp_decay_len/cp_floor: critical-period plasticity gating
        (CANDIDATE_MECHANISMS.md #D2, Achille et al. 2019-inspired). Scales
        A_PLUS/A_MINUS (in 'fep' and 'btsp' only) by 1.0 for the first
        cp_steps of the lifetime, then linearly decays to cp_floor over the
        next cp_decay_len steps. Defaults (cp_steps=1e9) leave every prior
        result unchanged. fep_punish_gate_steps: fep_punish cannot TRIGGER
        a new punishment burst during the lifetime's first N steps (default
        0 = unchanged) -- tests whether punishment *timing* rather than
        intensity explains the Task 11/12 bracket, since a critical-period
        account predicts early punishment is disproportionately damaging.

        homeo='mult' (default, unchanged from every prior task): every 500
        steps, scale a neuron's ENTIRE incoming weight vector by a common
        factor toward a target firing rate. Miller & MacKay (1994): this
        form provably CANNOT break initial left/right symmetry.
        homeo='sub': Miller & MacKay / Zenke & Gerstner-style subtractive
        normalization, every `homeo_every` steps (default 20, vs mult's
        500). Each postsynaptic neuron's total incoming EXCITATORY plastic
        weight is conserved at its lifetime-start value; whatever Hebbian
        potentiation added this interval is subtracted back out evenly
        across that neuron's incoming synapses, so growth in one synapse
        necessarily shrinks its neighbors -- the competitive form that CAN
        break symmetry, on a timescale close to the Hebbian update itself.

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
        chaining a second run() call from where the first left off).

        clamp_LR: "bench test" / open-loop diode probe. If given, a tuple
        (L_fixed, R_fixed) of (B,) tensors -- the smell L/R sensor injection
        is forced to these fixed values every step instead of being computed
        from food geometry, decoupling the network's sensor->motor response
        from any embodied confound (movement, food layout, wall reflex).
        Pair with freeze_motion=True (body position frozen, no movement
        update at all -- wall/food_close/nearest all stay constant too) and
        stdp_on=False (testing the CURRENT wiring's response, not letting it
        keep learning during the probe) for a clean causal test: does mL-mR
        track L_fixed-R_fixed the way a diode's output tracks its input,
        with nothing else in the loop able to explain the result. Default
        None leaves every prior result unchanged (verified no-op)."""
        dev, dt = self.device, self.dtype
        B = weight_plastic.shape[0]
        N = self.N
        wp = weight_plastic
        surprise = (learning == 'surprise')
        fep = (learning == 'fep')
        eh = (learning == 'eh')
        btsp = (learning == 'btsp')
        evolve_rule = (learning == 'evolverule')
        if evolve_rule:
            # Per-synapse Hebbian rates gathered once from the evolved
            # per-pool-pair genome (rule_a_plus/rule_a_minus, each (B,
            # n_pairs)) -- constant for the whole lifetime, unlike wp which
            # evolves every step. Same plain-Hebbian math as 'fep', just
            # A_PLUS/A_MINUS replaced by these gathered per-synapse values.
            a_plus_syn = rule_a_plus[:, self.pair_idx]
            a_minus_syn = rule_a_minus[:, self.pair_idx]
        eat_r2 = EAT_RADIUS2 if eat_radius2 is None else eat_radius2
        nf = N_FOOD if n_food is None else n_food
        if surprise:
            elig = torch.zeros(B, self.n_plastic, device=dev, dtype=dt)
            value = torch.zeros(B, device=dev, dtype=dt)
        if btsp:
            # Behavioural-timescale eligibility: same Hebbian-coincidence
            # trace math as 'surprise', but decayed far slower (~1000 steps
            # vs ~40) and applied to weights only at eat events, not every
            # step.
            elig = torch.zeros(B, self.n_plastic, device=dev, dtype=dt)
        if eh:
            # Exploratory Hebbian / reward-modulated Hebbian with double
            # baseline subtraction (Legenstein, Chase, Schwartz & Maass
            # 2010; Hoerzer, Legenstein & Maass 2014). dopa_bar/a_bar are
            # fast (eh_bar_decay, default 0.8 -> ~5-step horizon per the
            # paper) low-pass filters of reward and PER-NEURON postsynaptic
            # activity. Unlike every other rule here, [a_i - a_bar_i] is a
            # SIGNED per-neuron fluctuation-from-baseline term -- it can be
            # positive for mL and negative for mR on the same step, which is
            # the missing ingredient for building a differential sL->mL vs
            # sL->mR pathway (see CANDIDATE_MECHANISMS.md Part 1.1).
            dopa_bar = torch.zeros(B, device=dev, dtype=dt)
            a_bar = torch.zeros(B, N, device=dev, dtype=dt)
        if fep_punish:
            steps_since_food = torch.zeros(B, device=dev, dtype=dt)
            punish_timer = torch.zeros(B, device=dev, dtype=dt)

        sub_homeo = (homeo == 'sub')
        if sub_homeo:
            # Conserve each postsynaptic neuron's total incoming excitatory
            # plastic weight at this lifetime's starting value.
            target_post = torch.zeros(B, N, device=dev, dtype=dt)
            if self.post_exc.numel():
                target_post.index_add_(1, self.post_exc, wp[:, self.exc_mask])

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
        near_hist = [] if record_steering else None
        hits_hist = [] if record_steering else None
        wall_hist = [] if record_steering else None
        cx_hist = [] if record_steering else None
        cy_hist = [] if record_steering else None
        food_hist = [] if record_steering else None

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
            if clamp_LR is not None:
                L, R = clamp_LR[0], clamp_LR[1]
            wall = torch.clamp(1.0 - torch.minimum(torch.minimum(cx, cy),
                               torch.minimum(1.0 - cx, 1.0 - cy)) / 0.15, min=0.0)
            food_close = 1.0 / (1.0 + (nearest / 0.3) ** 2)
            hunger = 1.0 - energy_val / 100.0
            tired = torch.clamp(tiredness, max=1.0)

            if fep_punish:
                trigger = (wall > fep_wall_thresh) | (steps_since_food > fep_timeout_steps)
                if t < fep_punish_gate_steps:
                    # Critical-period test: punishment cannot START a new
                    # burst this early in the lifetime (an in-progress burst
                    # from before -- impossible at t=0 -- would still count
                    # down normally; this only blocks new triggers).
                    trigger = trigger & False
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
            # Critical-period gating (CANDIDATE_MECHANISMS.md #D2): 1.0 for
            # the first cp_steps, linear decay to cp_floor over the next
            # cp_decay_len steps, then held. Defaults leave this at 1.0
            # always (cp_steps=1e9). Only 'fep'/'btsp' consult it.
            if t < cp_steps:
                cp_mult = 1.0
            else:
                frac = min(1.0, (t - cp_steps) / max(1, cp_decay_len))
                cp_mult = 1.0 - frac * (1.0 - cp_floor)
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

            if stdp_on and evolve_rule:
                # Same plain, unmodulated Hebbian STDP as 'fep' (still
                # consults cp_mult, a no-op unless --cp-steps is also set),
                # but A_PLUS/A_MINUS are per-synapse values gathered from the
                # evolved per-pool-pair genome instead of global constants.
                pf = fired[:, self.post_p]
                qf = fired[:, self.pre_p]
                le = torch.minimum(energy[:, self.pre_p], energy[:, self.post_p])
                wp = wp + a_plus_syn * cp_mult * trace_pre[:, self.pre_p] * le * pf.to(dt)
                wp = wp - a_minus_syn * cp_mult * trace_post[:, self.post_p] * le * qf.to(dt)
                wp = torch.where(self.exc_mask.unsqueeze(0),
                                 torch.clamp(wp, 0.0, W_MAX),
                                 torch.clamp(wp, -W_MAX, 0.0))

            if stdp_on and not surprise and not fep and not eh and not btsp and not evolve_rule:
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
                # cp_mult (1.0 unless --cp-steps set) applies critical-period
                # gating: full plasticity early, decaying later.
                pf = fired[:, self.post_p]
                qf = fired[:, self.pre_p]
                le = torch.minimum(energy[:, self.pre_p], energy[:, self.post_p])
                wp = wp + A_PLUS * cp_mult * trace_pre[:, self.pre_p] * le * pf.to(dt)
                wp = wp - A_MINUS * cp_mult * trace_post[:, self.post_p] * le * qf.to(dt)
                wp = torch.where(self.exc_mask.unsqueeze(0),
                                 torch.clamp(wp, 0.0, W_MAX),
                                 torch.clamp(wp, -W_MAX, 0.0))

            if stdp_on and btsp:
                # Slow Hebbian eligibility (btsp_decay, ~1000-step horizon),
                # gated by cp_mult like 'fep'. Applied to WEIGHTS only on an
                # eat-event step (hits>0, computed later this same step and
                # looked up via a one-step-delayed hit mask is awkward here,
                # so instead: accumulate every step, and separately apply +
                # reset at the eat-detection point later in the loop -- see
                # "BTSP event-gated application" below.
                pf = fired[:, self.post_p]
                qf = fired[:, self.pre_p]
                le = torch.minimum(energy[:, self.pre_p], energy[:, self.post_p])
                hebb = (A_PLUS * cp_mult * trace_pre[:, self.pre_p] * pf.to(dt)
                       - A_MINUS * cp_mult * trace_post[:, self.post_p] * qf.to(dt)) * le
                elig = elig * btsp_decay + hebb

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

            if stdp_on and eh:
                # Delta_w_ij = eta * [R - Rbar] * x_j * [a_i - abar_i], using
                # the baselines from BEFORE this step's update (so the term
                # reflects "did reward beat its recent running average, and
                # did this postsynaptic neuron fire more/less than ITS
                # recent running average" -- both signed, both local).
                delta_r = (dopamine - dopa_bar).unsqueeze(1)          # (B,1)
                a_fluct = smoothed - a_bar                            # (B,N)
                x_j = smoothed[:, self.pre_p]                         # (B,n_plastic)
                a_fluct_i = a_fluct[:, self.post_p]                   # (B,n_plastic)
                wp = wp + eh_lr * delta_r * x_j * a_fluct_i
                wp = torch.where(self.exc_mask.unsqueeze(0),
                                 torch.clamp(wp, 0.0, W_MAX),
                                 torch.clamp(wp, -W_MAX, 0.0))
                dopa_bar = dopa_bar * eh_bar_decay + dopamine * (1.0 - eh_bar_decay)
                a_bar = a_bar * eh_bar_decay + smoothed * (1.0 - eh_bar_decay)

            if stdp_on:
                if not sub_homeo:
                    if (t + 1) % 500 == 0:
                        rate_post = firing_rate[:, self.post_p]
                        sf = torch.ones_like(rate_post)
                        sf = torch.where(rate_post < 0.02, torch.full_like(sf, 1.0 + 0.05 * HOMEO_STRENGTH), sf)
                        sf = torch.where(rate_post > 0.02, torch.full_like(sf, max(0.7, 1.0 - 0.05 * HOMEO_STRENGTH)), sf)
                        wp = wp * sf
                        wp = torch.where(self.exc_mask.unsqueeze(0),
                                         torch.clamp(wp, 0.0, W_MAX),
                                         torch.clamp(wp, -W_MAX, 0.0))
                else:
                    if (t + 1) % homeo_every == 0 and self.post_exc.numel():
                        # Subtractive normalization (Miller & MacKay 1994):
                        # conserve each postsynaptic neuron's total incoming
                        # excitatory plastic weight at its lifetime-start
                        # value. Whatever Hebbian potentiation added this
                        # interval is subtracted back out evenly across that
                        # neuron's incoming synapses -- growth in one synapse
                        # necessarily competes with its neighbors, which is
                        # the form that CAN break symmetry (unlike the
                        # multiplicative 'mult' branch above).
                        cur_post = torch.zeros(B, N, device=dev, dtype=dt)
                        cur_post.index_add_(1, self.post_exc, wp[:, self.exc_mask])
                        excess = cur_post - target_post  # (B, N)
                        corr = excess[:, self.post_exc] / self.fan_in_exc[self.post_exc].unsqueeze(0)
                        wp_exc = wp[:, self.exc_mask] - corr
                        wp = wp.clone()
                        wp[:, self.exc_mask] = wp_exc
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
                near_hist.append(nearest.detach())
                wall_hist.append(wall.detach())
                cx_hist.append(cx.detach())
                cy_hist.append(cy.detach())
                food_hist.append(food.detach().clone())
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
            if freeze_motion:
                new_head, new_cx, new_cy = head, cx, cy

            av_f = alive.to(dt)
            head = torch.where(alive, new_head, head)
            cx = torch.where(alive, new_cx, cx)
            cy = torch.where(alive, new_cy, cy)
            head_vel = head_vel * av_f

            dxh = food[:, :, 0] - cx.unsqueeze(1)
            dyh = food[:, :, 1] - cy.unsqueeze(1)
            hit = (dxh * dxh + dyh * dyh < eat_r2) & alive.unsqueeze(1)
            hits = hit.sum(dim=1).to(dt)
            if record_steering:
                hits_hist.append(hits.detach())

            if stdp_on and btsp:
                # BTSP event-gated application: on rows that just ate,
                # apply the accumulated slow trace to weights and reset it,
                # so credit is assigned once per approach-and-eat rather
                # than smeared continuously (and never during a fep_punish
                # window, since those trigger on wall/timeout, not on
                # eating -- sidesteps Task 10's interference mechanism).
                ate = (hits > 0).unsqueeze(1).to(dt)  # (B,1)
                wp = wp + btsp_lr * elig * ate
                wp = torch.where(self.exc_mask.unsqueeze(0),
                                 torch.clamp(wp, 0.0, W_MAX),
                                 torch.clamp(wp, -W_MAX, 0.0))
                elig = elig * (1.0 - ate)

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
            out['nearest'] = torch.stack(near_hist, dim=0)   # (T, B), distance to nearest food
            out['hits'] = torch.stack(hits_hist, dim=0)      # (T, B), food-eat events per step
            out['wall'] = torch.stack(wall_hist, dim=0)      # (T, B), wall-proximity signal (0-1)
            out['cx'] = torch.stack(cx_hist, dim=0)          # (T, B), creature x position (pre-move-of-this-step)
            out['cy'] = torch.stack(cy_hist, dim=0)          # (T, B), creature y position
            out['food_pos'] = torch.stack(food_hist, dim=0)  # (T, B, nf, 2), food positions (pre-eat-of-this-step)
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


def long_range_correlation(dL, nearest, W=150):
    """Task 14 follow-up: outcome-based, long-horizon steering metric.
    steering_correlation measures instantaneous linear coupling between
    sensor asymmetry and motor asymmetry -- it's blind to mechanisms that
    achieve real approach through an indirect or delayed route (speed
    modulation, multi-step heading correction), which candidate #4's
    champions turned out to use (confirmed via a food-doesn't-move-controlled
    trajectory analysis: net distance-to-food closed over 200-step windows
    ending at eat events was strongly positive vs. ~zero for random windows
    in the same trajectory, even though instantaneous steering_correlation
    was near zero or negative for those same champions).

    dL: (T, B) sensor L-R asymmetry (same as steering_correlation).
    nearest: (T, B) distance to nearest food each step.
    W: window length in steps.

    Returns per-batch Pearson r between dL(t) and the NET DISTANCE CLOSED
    over the following W steps (nearest[t] - nearest[t+W], positive =
    got closer) -- directly measures "does stronger smell asymmetry now
    predict real approach progress over the next W steps", independent of
    *how* the network achieves it. Caveat: food teleports to a new random
    location on each eat event, so windows spanning an eat event mix two
    different food targets -- a real but bounded source of noise this
    metric doesn't correct for."""
    T = dL.shape[0]
    if T <= W:
        return np.full(dL.shape[1], np.nan)
    delta_near = (nearest[:-W] - nearest[W:]).detach()   # (T-W, B), + = closer
    dl = dL[:-W].detach()
    dl = dl - dl.mean(dim=0, keepdim=True)
    dn = delta_near - delta_near.mean(dim=0, keepdim=True)
    num = (dl * dn).sum(dim=0)
    den = torch.sqrt((dl * dl).sum(dim=0) * (dn * dn).sum(dim=0)) + 1e-9
    return (num / den).cpu().numpy()


def hunt_score(nearest, hits, W=200, seed=0):
    """Task 14 follow-up, eat-event-conditioned version. long_range_correlation
    correlates sensor asymmetry against future approach at every single step,
    which turned out too diluted by non-approach time to pick up the real
    signal. This instead asks the direct behavioral question: over the W
    steps immediately before each successful eat, how much closer did the
    creature get to that food, compared to a food-doesn't-move-controlled
    null (the same trajectory's own random W-step windows, unrelated to any
    eat event -- an undirected walk should show ~zero net approach there,
    so this isolates genuine directed approach from the trivial "must be
    close to eat" effect a naive pre-eat-only check would conflate with it).

    nearest, hits: (T, B) torch tensors from a record_steering=True run.
    Returns (B,) numpy: mean(pre-eat net approach) - mean(null net approach),
    in arena-distance units (0-1 scale). Positive = real approach signal
    beyond chance; ~0 = no directed approach detectable at this timescale."""
    near = nearest.detach().cpu().numpy()
    hit_arr = hits.detach().cpu().numpy()
    T, B = near.shape
    rng = np.random.default_rng(seed)
    out = np.full(B, np.nan)
    for b in range(B):
        eat_steps = np.flatnonzero(hit_arr[:, b] > 0)
        pre = [near[hi - W, b] - near[hi - 1, b] for hi in eat_steps if hi - W > 0]
        if not pre:
            continue
        n_null = max(len(pre), 1) * 5
        starts = rng.integers(0, max(1, T - W), size=n_null)
        null = [near[s, b] - near[s + W - 1, b] for s in starts]
        out[b] = float(np.mean(pre) - np.mean(null))
    return out


def hunt_score_v2(nearest, hits, wall, W=200, wall_thresh=0.15, seed=0):
    """hunt_score confounded control (Task 14 follow-up part 2): a null of
    RANDOM, UNTRAINED, unlearned weights scored the same as every evolved
    champion on hunt_score (~0.13-0.18 for all). Root cause: the environment
    itself has a hardcoded wall-avoidance reflex baked into the physics loop
    (turn += dh_wall * wall * 0.3, active regardless of the evolved/learned
    circuit) that steers the creature toward the arena center whenever it's
    near a wall -- since food is spawned uniformly, this alone produces a
    center-ward drift that looks like "approaching food" in hunt_score,
    with zero learning involved.

    This version restricts every window (both the pre-eat sample and its
    null) to ones where wall-proximity stayed below wall_thresh throughout
    -- i.e. periods where the hardcoded reflex is inactive, so any residual
    approach signal has to come from the evolved/learned circuit instead.
    Same (B,) numpy output and sign convention as hunt_score; NaN where a
    seed has no qualifying pre-eat window."""
    near = nearest.detach().cpu().numpy()
    hit_arr = hits.detach().cpu().numpy()
    wall_arr = wall.detach().cpu().numpy()
    T, B = near.shape
    rng = np.random.default_rng(seed)
    out = np.full(B, np.nan)
    for b in range(B):
        eat_steps = np.flatnonzero(hit_arr[:, b] > 0)
        pre = []
        for hi in eat_steps:
            if hi - W > 0 and wall_arr[hi - W:hi, b].max() < wall_thresh:
                pre.append(near[hi - W, b] - near[hi - 1, b])
        if not pre:
            continue
        n_null_target = max(len(pre), 1) * 5
        null = []
        attempts = 0
        while len(null) < n_null_target and attempts < n_null_target * 20:
            s = rng.integers(0, max(1, T - W))
            attempts += 1
            if wall_arr[s:s + W, b].max() < wall_thresh:
                null.append(near[s, b] - near[s + W - 1, b])
        if not null:
            continue
        out[b] = float(np.mean(pre) - np.mean(null))
    return out


# Diode bench test (Task 17): the standard diagnostic going forward, per
# Dave's "microcontroller + testable diodes" framing. Every prior behavioral
# metric (approach_frac, long_range_correlation, hunt_score, hunt_score_v2)
# observed the network embedded in the full closed loop and tried to infer
# causation from correlation -- every one either measured a tautology or
# failed a null control (Task 16). This instead clamps the smell L/R sensor
# injection to a fixed value, freezes the body (no movement, so food/wall/
# hunger can't drift either), turns learning off, and reads steady-state
# mL-mR -- a direct, isolated, causal read of "does this wiring turn toward
# a controlled input the way a diode's output tracks its input," with
# nothing else in the loop able to explain the result. Decomposes the
# response into BIAS (turn tendency at symmetric input, a=0 -- a built-in
# spin/circle tendency unrelated to sensing) and SLOPE (how much the turn
# actually changes per unit of sensor asymmetry -- the thing genuine
# steering requires). First use (Task 17) found candidate #4's expressed
# weights carry a large, nearly input-independent BIAS (~0.05-0.12) and
# only a small SLOPE -- explaining its high food count (a fixed spiral
# sweeps more arena) and near-zero steering-r (a constant bias doesn't
# track live sensor asymmetry) at once.
ASYM_SWEEP_DEFAULT = (-1.0, -0.5, 0.0, 0.5, 1.0)


def bench_diode(sim, wp, bench_kwargs, asym_sweep=ASYM_SWEEP_DEFAULT, T=800, settle=200,
                 n_trials=3, seed0=1000):
    """wp: (1, n_plastic) weight snapshot to test (RAW genome or EXPRESSED
    post-lifetime weights). bench_kwargs: dict of run() kwargs identifying
    the learning mode (e.g. dict(learning='fep', fep_punish=False), or
    dict(learning='evolverule', rule_a_plus=..., rule_a_minus=...)) --
    stdp_on is always False here regardless, since this tests the CURRENT
    wiring's response, not further learning. Returns dict(asym, response,
    bias, slope, corr) -- bias/slope are the intercept/slope of a linear
    fit of response vs asym; corr is the Pearson r of that fit."""
    responses = []
    for a in asym_sweep:
        L = torch.full((1,), (1 + a) / 2, dtype=sim.dtype, device=sim.device)
        R = torch.full((1,), (1 - a) / 2, dtype=sim.dtype, device=sim.device)
        trial_resp = []
        for trial in range(n_trials):
            torch.manual_seed(seed0 + trial)
            out = sim.run(wp.clone(), T, reflex_scale=0.0, stdp_on=False, record_steering=True,
                          clamp_LR=(L, R), freeze_motion=True, **bench_kwargs)
            trial_resp.append(out['dM'][-settle:, 0].mean().item())
        responses.append(float(np.mean(trial_resp)))
    asym = np.array(asym_sweep)
    resp = np.array(responses)
    slope, bias = np.polyfit(asym, resp, 1)
    corr = float(np.corrcoef(asym, resp)[0, 1]) if resp.std() > 1e-9 else 0.0
    return dict(asym=asym.tolist(), response=resp.tolist(), bias=float(bias),
                slope=float(slope), corr=corr)


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


# --------------------------------------------------------------------------- #
# Task 13 (CANDIDATE_MECHANISMS.md #0): asym weight-space diagnostic.
# mean(w[sL->mL]) - mean(w[sL->mR]) and the sR mirror, computed directly on
# a genome (numpy, plastic-synapse-indexed). Measures the thing the learning
# mechanism is supposed to build (a differential pathway), rather than
# inferring it from noisy behavioral steering correlation.
# --------------------------------------------------------------------------- #
def build_asym_masks(scaffold):
    plastic = ~scaffold['is_reflex']
    pre_p = scaffold['pre'][plastic]
    post_p = scaffold['post'][plastic]
    sL0, sL1 = POOLS['sL']; sR0, sR1 = POOLS['sR']
    mL0, mL1 = POOLS['mL']; mR0, mR1 = POOLS['mR']
    return dict(
        sL_mL=(pre_p >= sL0) & (pre_p < sL1) & (post_p >= mL0) & (post_p < mL1),
        sL_mR=(pre_p >= sL0) & (pre_p < sL1) & (post_p >= mR0) & (post_p < mR1),
        sR_mR=(pre_p >= sR0) & (pre_p < sR1) & (post_p >= mR0) & (post_p < mR1),
        sR_mL=(pre_p >= sR0) & (pre_p < sR1) & (post_p >= mL0) & (post_p < mL1),
    )


def compute_asym(genome, masks):
    def m(mask):
        return float(genome[mask].mean()) if mask.any() else 0.0
    asym_L = m(masks['sL_mL']) - m(masks['sL_mR'])
    asym_R = m(masks['sR_mR']) - m(masks['sR_mL'])
    return asym_L, asym_R


# --------------------------------------------------------------------------- #
# Task 13 candidate #4 (CANDIDATE_MECHANISMS.md #4, Najarro & Risi 2020):
# evolve-the-plasticity-rule. Instead of evolving one value per plastic
# synapse (~n_plastic genes, structurally able to memorize a layout-specific
# weight pattern), evolve one Hebbian rate PAIR per (pre_pool, post_pool)
# combination actually present in the topology (a couple hundred genes at
# most) -- every synapse in the same pool-pair shares the same evolved
# A_PLUS/A_MINUS, and the lifetime's own Hebbian dynamics (same math as
# 'fep') expresses those rates into synaptic weights from a fixed, shared
# starting point. Directly shrinks what evolution can overfit to.
# --------------------------------------------------------------------------- #
def pool_of_index(idx, pools):
    for name, (lo, hi) in pools.items():
        if lo <= idx < hi:
            return name
    return 'hid'


def build_pair_map(scaffold):
    """Returns (pair_idx, n_pairs, pair_labels). pair_idx is a (n_plastic,)
    int array mapping each plastic synapse to an index into a de-duplicated
    list of (pre_pool, post_pool) pairs actually present in this topology
    (pair_labels, same order)."""
    plastic = ~scaffold['is_reflex']
    pre_p = scaffold['pre'][plastic]
    post_p = scaffold['post'][plastic]
    pre_pool = [pool_of_index(int(i), POOLS) for i in pre_p]
    post_pool = [pool_of_index(int(i), POOLS) for i in post_p]
    pairs = sorted(set(zip(pre_pool, post_pool)))
    pair_to_idx = {p: i for i, p in enumerate(pairs)}
    pair_idx = np.array([pair_to_idx[(a, b)] for a, b in zip(pre_pool, post_pool)], dtype=np.int64)
    return pair_idx, len(pairs), pairs


# --------------------------------------------------------------------------- #
# Task 13 candidate A (evo-devo guidance-code wiring, Eph/ephrin-inspired):
# instead of evolving a plasticity rate per pool-pair (#4), evolve a tiny
# per-POOL "guidance code" vector (analogous to a receptor/ligand expression
# level). Two candidate synapses connect strongly at the START of a lifetime
# only if their pre/post pools' codes are compatible (high dot product) --
# a compact, evolved, one-shot DEVELOPMENTAL gate applied before plasticity
# ever runs, not a hardcoded L/R rule: evolution has to discover which pools'
# codes should end up complementary, same as real axon guidance discovers
# which growth cones follow which gradients. Topology (the candidate edge
# list) stays exactly as built by build_scaffold_numpy/K-NN -- only the
# per-synapse INITIAL weight magnitude (sign preserved) is gated, keeping
# the batched-tensor architecture: one shared edge list, per-individual gate.
# --------------------------------------------------------------------------- #
POOL_NAMES = list(POOLS.keys()) + ['hid']
POOL_NAME_TO_ID = {name: i for i, name in enumerate(POOL_NAMES)}
N_POOLS = len(POOL_NAMES)


def build_pool_id(N):
    """(N,) int array: every neuron index -> its pool id (POOL_NAMES order)."""
    return np.array([POOL_NAME_TO_ID[pool_of_index(i, POOLS)] for i in range(N)], dtype=np.int64)


# Task 17 lateral-bias structural genes (see BatchSim.gate_weights docstring)
LAT_PAIRS = [('sL', 'mL'), ('sL', 'mR'), ('sR', 'mL'), ('sR', 'mR')]
LAT_PAIR_TO_ID = {p: i for i, p in enumerate(LAT_PAIRS)}


def build_lat_idx(pre_pool_id_np, post_pool_id_np):
    """(n_plastic,) int array: index into LAT_PAIRS (0-3) if this synapse's
    (pre_pool, post_pool) is one of the 4 sensor<->motor pairs, else -1."""
    id_to_name = {v: k for k, v in POOL_NAME_TO_ID.items()}
    idx = np.full(len(pre_pool_id_np), -1, dtype=np.int64)
    for i in range(len(pre_pool_id_np)):
        key = (id_to_name[int(pre_pool_id_np[i])], id_to_name[int(post_pool_id_np[i])])
        if key in LAT_PAIR_TO_ID:
            idx[i] = LAT_PAIR_TO_ID[key]
    return idx


def run_lifetime(sim, weight_plastic, args, reflex_scale, T, record_steering=False):
    """One genome-batch's full lifetime: optional bootcamp phase (easy,
    dense-reward, annealed exploration -- shapes weights) followed by the
    real (scored) phase at normal difficulty. Only the real phase's
    food_eaten/steering counts for fitness -- bootcamp food doesn't.

    For --learning evolverule, `weight_plastic` is NOT a per-synapse weight
    tensor -- it's the evolved genome batch (B, 2*n_pairs) of per-pool-pair
    Hebbian rates. The actual synaptic weights (wp) start instead from the
    topology's own fixed init_weight (shared, non-evolved), and the rates
    are gathered per-synapse inside sim.run() via rule_a_plus/rule_a_minus."""
    evolve_rule = (args.learning == 'evolverule')
    wiring_guide = (args.wiring in ('guide', 'guide_lat'))
    wiring_lat = (args.wiring == 'guide_lat')
    rule_kw = {}
    if evolve_rule:
        genome_batch = weight_plastic
        B = genome_batch.shape[0]
        wp = sim.init_weight.unsqueeze(0).expand(B, -1).clone()
        rule_kw = dict(rule_a_plus=genome_batch[:, :sim.n_pairs],
                       rule_a_minus=genome_batch[:, sim.n_pairs:])
    elif wiring_guide:
        genome_batch = weight_plastic
        B = genome_batch.shape[0]
        n_code_genes = sim.n_pools * args.guide_code_dim
        guide_code = genome_batch[:, :n_code_genes].view(B, sim.n_pools, -1)
        lat_bias = genome_batch[:, n_code_genes:] if wiring_lat else None
        wp = sim.gate_weights(sim.init_weight, guide_code, gain=args.guide_gain, lat_bias=lat_bias)
    else:
        wp = weight_plastic
    fep_kw = dict(fep_punish=args.fep_punish, fep_punish_t=args.fep_punish_t,
                  fep_wall_thresh=args.fep_wall_thresh, fep_timeout_steps=args.fep_timeout_steps,
                  fep_punish_gate_steps=args.fep_punish_gate_steps)
    homeo_kw = dict(homeo=args.homeo, homeo_every=args.homeo_every)
    eh_kw = dict(eh_lr=args.eh_lr, eh_bar_decay=args.eh_bar_decay)
    cp_kw = dict(cp_steps=args.cp_steps, cp_decay_len=args.cp_decay_len, cp_floor=args.cp_floor)
    btsp_kw = dict(btsp_decay=args.btsp_decay, btsp_lr=args.btsp_lr)
    if args.bootcamp:
        boot_out = sim.run(wp, args.boot_t, reflex_scale, stdp_on=True,
                           learning=args.learning, elig_decay=args.elig_decay,
                           elig_lr=args.elig_lr, value_lr=args.value_lr,
                           eat_radius2=args.boot_eat_r2, n_food=args.boot_n_food,
                           food_spawn_radius=args.boot_food_radius,
                           metab_scale=args.boot_metab_scale,
                           motor_noise_start=args.boot_noise_start,
                           motor_noise_end=args.boot_noise_end,
                           use_eye=args.use_eye, **fep_kw, **homeo_kw, **eh_kw,
                           **cp_kw, **btsp_kw, **rule_kw)
        wp = boot_out['weight_plastic']
    return sim.run(wp, T, reflex_scale, stdp_on=True, record_steering=record_steering,
                   learning=args.learning, elig_decay=args.elig_decay,
                   elig_lr=args.elig_lr, value_lr=args.value_lr, use_eye=args.use_eye,
                   **fep_kw, **homeo_kw, **eh_kw, **cp_kw, **btsp_kw, **rule_kw)


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
    ap.add_argument('--learning', type=str, default='stdp',
                    choices=['stdp', 'surprise', 'fep', 'eh', 'btsp', 'evolverule'],
                    help='stdp = reward-gated instantaneous STDP; surprise = eligibility x TD-error; '
                         'fep = plain unmodulated Hebbian STDP, pair with --fep-punish; '
                         'eh = exploratory Hebbian, signed postsynaptic-fluctuation x reward-fluctuation '
                         "(CANDIDATE_MECHANISMS.md #1); btsp = event-gated behavioural-timescale "
                         'eligibility, applied only on eat events (CANDIDATE_MECHANISMS.md #3/B2); '
                         'evolverule = evolve per-pool-pair Hebbian rates instead of per-synapse '
                         'weights (CANDIDATE_MECHANISMS.md #4, Najarro & Risi)')
    ap.add_argument('--rule-rate-max', type=float, default=0.05,
                    help="learning='evolverule' only: upper bound on evolved per-pair A_PLUS/A_MINUS "
                         '(genes init at the fep defaults, 0.008/0.010)')
    ap.add_argument('--wiring', type=str, default='none', choices=['none', 'guide', 'guide_lat'],
                    help='none = unchanged (per-synapse weight genome, as picked by --learning). '
                         'guide = evo-devo guidance-code wiring (CANDIDATE_MECHANISMS.md #A, '
                         'Eph/ephrin-inspired): genome becomes a tiny per-POOL code vector instead '
                         'of per-synapse weights; initial synaptic weight = fixed topology magnitude '
                         '* sigmoid(gain * code_pre . code_post), applied once before the lifetime, '
                         "then --learning plasticity runs on top as normal. guide_lat (Task 17, Dave's "
                         '"solid starting structure" -- contralateral wiring bias): same as guide, '
                         'plus 4 extra evolved genes, one per sensor<->motor pool pair (sL-mL, sL-mR, '
                         'sR-mL, sR-mR), giving evolution a direct low-dimensional knob for exactly '
                         'the structural degree of freedom real nervous systems specialize '
                         '(decussation), instead of hoping the ambient per-pool code formula finds it. '
                         'Not combinable with --learning evolverule (both replace the weight-init '
                         "source).")
    ap.add_argument('--guide-code-dim', type=int, default=3,
                    help="--wiring guide only: dimensionality of each pool's evolved guidance code")
    ap.add_argument('--guide-gain', type=float, default=4.0,
                    help="--wiring guide only: sigmoid gain on the code dot-product "
                         '(higher = sharper connect/prune boundary)')
    ap.add_argument('--fitness-r-weight', type=float, default=0.0,
                    help='if > 0, TRAINING fitness = food + this * max(0, steering_r) instead of '
                         'food alone -- selects directly for positive linear L-R steering '
                         'correlation, not just food count (Task 14 follow-up: candidate #4 roughly '
                         "doubled food but didn't reliably build positive steering; this pushes "
                         'selection toward the metric this project actually cares about). Default '
                         '0.0 = unchanged (food-only fitness, every prior task).')
    ap.add_argument('--long-window', type=int, default=150,
                    help='window (steps) for the long-range steering correlation reported '
                         'alongside the standard (instantaneous) one on the held-out champion -- '
                         'correlates sensor L-R asymmetry now against NET distance-to-food closed '
                         'over the following W steps, catching delayed/indirect approach the '
                         'instantaneous metric misses (Task 14 follow-up).')
    ap.add_argument('--elig-decay', type=float, default=0.995)
    ap.add_argument('--elig-lr', type=float, default=0.03)
    ap.add_argument('--value-lr', type=float, default=0.02)
    ap.add_argument('--eh-lr', type=float, default=0.05, help='learning='"'"'eh'"'"' only: weight-update rate')
    ap.add_argument('--eh-bar-decay', type=float, default=0.8,
                    help="learning='eh' only: low-pass decay for the reward/activity running "
                         "baselines (0.8 -> ~5-step horizon, matching Hoerzer et al. 2014)")
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
    ap.add_argument('--homeo', type=str, default='mult', choices=['mult', 'sub'],
                    help="mult = original multiplicative homeostasis, every 500 steps "
                         "(default, reproduces every prior task unchanged). sub = Miller & "
                         "MacKay-style subtractive normalization of each neuron's total "
                         "incoming excitatory plastic weight, every --homeo-every steps -- "
                         "the form that can break L/R symmetry (CANDIDATE_MECHANISMS.md #2)")
    ap.add_argument('--homeo-every', type=int, default=20,
                    help='steps between --homeo sub applications (ignored for mult, which is '
                         'hardcoded to 500 to keep old results reproducible)')
    ap.add_argument('--fep-punish-gate-steps', type=int, default=0,
                    help='fep_punish cannot trigger a NEW burst during the first N steps of the '
                         'lifetime (default 0 = unchanged). Tests punishment TIMING vs intensity '
                         '(CANDIDATE_MECHANISMS.md #D2)')
    ap.add_argument('--cp-steps', type=int, default=10**9,
                    help="critical-period gating (learning in {fep,btsp} only): full plasticity "
                         "for the lifetime's first N steps, then decays toward --cp-floor over "
                         "--cp-decay-len steps. Default (1e9) = always full plasticity, unchanged.")
    ap.add_argument('--cp-decay-len', type=int, default=1,
                    help='steps over which plasticity decays from 1.0 to --cp-floor after --cp-steps')
    ap.add_argument('--cp-floor', type=float, default=1.0,
                    help='plasticity multiplier floor after the critical period decays (1.0 = no-op)')
    ap.add_argument('--btsp-decay', type=float, default=0.999,
                    help="learning='btsp' only: eligibility trace decay (0.999 -> ~1000-step horizon)")
    ap.add_argument('--btsp-lr', type=float, default=0.05,
                    help="learning='btsp' only: weight-update rate applied at each eat event")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f'device: {device}', flush=True)
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    print(f'learning: {args.learning}  dense_k: {args.dense_k}  homeo: {args.homeo}', flush=True)

    scaffold = build_scaffold_numpy(args.N, args.K, fi=0.2, seed=args.scaffold_seed, dense_k=args.dense_k)
    sim = BatchSim(scaffold, device)
    asym_masks = build_asym_masks(scaffold)
    evolve_rule = (args.learning == 'evolverule')
    wiring_guide = (args.wiring in ('guide', 'guide_lat'))
    wiring_lat = (args.wiring == 'guide_lat')
    if evolve_rule and wiring_guide:
        raise SystemExit("--learning evolverule and --wiring guide/guide_lat both replace the "
                          "weight-init source and are not combinable yet -- pick one.")

    if wiring_guide:
        n_code_genes = sim.n_pools * args.guide_code_dim
        n_lat_genes = 4 if wiring_lat else 0
        n_genes = n_code_genes + n_lat_genes
        print(f'genes (guidance codes, {sim.n_pools} pools x {args.guide_code_dim}-dim'
              + (f' + {n_lat_genes} lateral-bias)' if wiring_lat else ')') + f': {n_genes}', flush=True)
        lo = np.concatenate([np.full(n_code_genes, -3.0), np.full(n_lat_genes, -5.0)])
        hi = np.concatenate([np.full(n_code_genes, 3.0), np.full(n_lat_genes, 5.0)])
        # neutral start: gate=0.5 everywhere for both codes AND the lateral-
        # bias genes -- evolution has to discover any structure, crossed or
        # uncrossed, from a flat prior; nothing here says which is correct.
        base_genome = np.zeros(n_genes)
    elif evolve_rule:
        n_genes = 2 * sim.n_pairs
        print(f'genes (plasticity-rule coefficients, {sim.n_pairs} pool-pairs): {n_genes}', flush=True)
        lo = np.zeros(n_genes)
        hi = np.full(n_genes, args.rule_rate_max)
        base_genome = np.concatenate([np.full(sim.n_pairs, A_PLUS), np.full(sim.n_pairs, A_MINUS)])
    else:
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

        use_r_fitness = args.fitness_r_weight > 0
        out = run_lifetime(sim, batch_w, args, reflex_scale, args.train_t,
                           record_steering=use_r_fitness)
        food = out['food_eaten'].view(P, S).mean(dim=1).cpu().numpy()
        if use_r_fitness:
            # Task 14 follow-up: select directly on (food + bonus for
            # positive linear steering correlation), not food alone -- #4
            # roughly doubled food without reliably building positive r;
            # this pushes selection toward the metric the project cares
            # about instead of letting it wander to whatever high-food
            # strategy it finds first.
            r_train = steering_correlation(out['dL'], out['dM']).reshape(P, S).mean(axis=1)
            fitness = food + args.fitness_r_weight * np.clip(r_train, 0.0, None)
        else:
            r_train = None
            fitness = food

        order = np.argsort(-fitness)
        ranked = [pop[i] for i in order]
        scored_food = food[order]
        best_fit = float(fitness[order[0]])
        mean_fit = float(fitness.mean())
        if best_fit > champion_fit:
            champion_fit = best_fit
            champion = ranked[0].copy()

        if evolve_rule or wiring_guide:
            # Genome here is rule coefficients or guidance codes, not
            # weights -- asym has to be measured on the EXPRESSED weights
            # this individual's own Hebbian dynamics produced this lifetime,
            # not the genome itself.
            wp_expressed = out['weight_plastic'][order[0] * S].detach().cpu().numpy()
            asym_L, asym_R = compute_asym(wp_expressed, asym_masks)
        else:
            asym_L, asym_R = compute_asym(ranked[0], asym_masks)
        r_note = f'  r_train {r_train[order[0]]:+.4f}' if use_r_fitness else ''
        print(f'gen {gen:3d}  reflex_scale {reflex_scale:.2f}  best {best_fit:6.2f}  '
              f'mean {mean_fit:6.2f}  champion-so-far {champion_fit:6.2f}  '
              f'food {scored_food[0]:6.2f}{r_note}  '
              f'asym_L {asym_L:+.4f}  asym_R {asym_R:+.4f}  '
              f't={time.time()-t0:6.1f}s', flush=True)

        pop = breed_next_generation(ranked, lo, hi, args.pop, args.elite_frac,
                                    args.mut_sigma, args.mut_rate, args.crossover_p, rng)

    np.save(args.out, champion)
    print(f'\nchampion saved to {args.out}', flush=True)

    if wiring_lat:
        lat_genes = champion[-4:]
        for (pn, qn), v in zip(LAT_PAIRS, lat_genes):
            print(f'champion lateral-bias gene {pn}->{qn}: {v:+.4f}', flush=True)

    if not (evolve_rule or wiring_guide):
        champ_asym_L, champ_asym_R = compute_asym(champion, asym_masks)
        print(f'champion asym_L (sL->mL minus sL->mR): {champ_asym_L:+.4f}')
        print(f'champion asym_R (sR->mR minus sR->mL): {champ_asym_R:+.4f}')

    # Held-out evaluation at reflex_scale=0.0 (the real no-reflex test) plus a
    # steering-correlation check on the same held-out champion.
    champ_t = torch.tensor(champion, dtype=torch.float32, device=device).unsqueeze(0)
    champ_batch = champ_t.repeat(args.heldout_seeds, 1)
    out = run_lifetime(sim, champ_batch, args, reflex_scale=0.0, T=args.heldout_t, record_steering=True)
    if evolve_rule or wiring_guide:
        # Asym on the champion's EXPRESSED weights from this held-out
        # lifetime (genome is rule coefficients, not weights -- see above).
        wp_expressed = out['weight_plastic'][0].detach().cpu().numpy()
        champ_asym_L, champ_asym_R = compute_asym(wp_expressed, asym_masks)
        print(f'champion asym_L (sL->mL minus sL->mR, expressed weights): {champ_asym_L:+.4f}')
        print(f'champion asym_R (sR->mR minus sR->mL, expressed weights): {champ_asym_R:+.4f}')
    foods = out['food_eaten'].cpu().numpy()
    r = steering_correlation(out['dL'], out['dM'])
    r_long = long_range_correlation(out['dL'], out['nearest'], W=args.long_window)
    print(f'\nFINAL held-out ({args.heldout_seeds} seeds, T={args.heldout_t}, reflex=0): '
          f'food {foods.mean():.2f} +/- {foods.std()/math.sqrt(len(foods)):.2f}')
    print(f'per-seed food: {list(foods)}')
    print(f'steering correlation (smell), per-seed: {list(np.round(r, 3))}')
    print(f'steering correlation (smell), mean: {r.mean():.3f}')
    print(f'long-range steering correlation (smell, W={args.long_window}), per-seed: {list(np.round(r_long, 3))}')
    print(f'long-range steering correlation (smell, W={args.long_window}), mean: {r_long.mean():.3f}')
    hunt = hunt_score(out['nearest'], out['hits'], W=args.long_window)
    print(f'hunt_score (eat-event-conditioned net approach, W={args.long_window}), per-seed: '
          f'{list(np.round(hunt, 3))}')
    print(f'hunt_score (eat-event-conditioned net approach, W={args.long_window}), mean: {np.nanmean(hunt):.3f}')
    if args.use_eye:
        r_eye = steering_correlation(out['dEye'], out['dM'])
        print(f'steering correlation (EYE), per-seed: {list(np.round(r_eye, 3))}')
        print(f'steering correlation (EYE), mean: {r_eye.mean():.3f}')
    print(f'oracle ceiling: {ceiling:.1f}')


if __name__ == '__main__':
    main()
