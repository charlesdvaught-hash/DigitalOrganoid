# Conduction + motor fix — results

Tested against `creature_embodied_rstdp.py` (`run_experiments.py`), 6–8 seeds,
4000–6000 steps, N=600 K=10. Applied identically to `index.html` and
`jules/organoid_simulator/creature_embodied.py`.

## Task 1 — conduction

`SYN_GAIN` (new constant) scales `weight × smoothedActivity[pre]` only, separate
from `W_max`/`targetG`. Swept 1–1000; picked 100.

| SYN_GAIN | rate | branching σ | steering r |
|---|---|---|---|
| 1 (old) | 0.0038 | 0.71 | −0.02 |
| 60 | 0.014 | 0.86 | +0.15 |
| 100 (chosen) | 0.027 | 0.94 | +0.16 |
| 300 | 0.27 | 1.00 | +0.12 (numeric overflow risk) |

100 sits near criticality (σ≈0.94) at a few-% duty cycle, well clear of the
overflow onset at 300.

## Task 2 — motor scaling

`MOTOR_SPEED_SCALE` replaces the `0.007` coefficient in
`speed = 0.0009 + K·fwd·Mg`. Swept 0.007–0.5; picked 0.10.

| MOTOR_SPEED_SCALE | food | mean speed |
|---|---|---|
| 0.007 (old) | 4.0 | 0.0013 |
| 0.10 (chosen) | 21.5 | 0.0059 (near optimum 0.0087) |
| 0.26+ | ≤0.5 | overshoots, destabilizes steering |

## Task 3 — gate

Steering correlation **−0.017 → +0.175**. Clearly positive. Gate passed.

Food eaten: **1.25 → 20.75 ± 3.24** (oracle ceiling 26.8, seed-matched) — 77%
of ceiling, vs 5% before.

## Task 4 — the real test: does it survive without the reflex arcs?

Deleted `sL→mL`, `sR→mR`, `sW→mL/mR` (reflex_scale=0) with fixes 1–2 in place:

| condition | food | steering r |
|---|---|---|
| reflex ×1.0 | 20.75 ± 3.24 | +0.175 |
| reflex ×0.25 | 12.00 ± 0.77 | — |
| reflex ×0.0 (deleted) | 3.25 ± 0.58 | −0.004 |

**No.** Competence collapses and steering correlation returns to ~0 once the
innate links are gone. The network now conducts and the body now moves at a
useful speed, but the recurrent, plastic wiring has not itself learned a
sensor→motor mapping in a single ~6000-step lifetime under vanilla
reward-modulated STDP. The reflex arcs are still doing 100% of the functional
steering — same automaton problem the handoff flagged, just now running on top
of a brain that can finally hear itself think.

Not patched around — no shaping, hardcoding, or scripted fallback added here.
This is the next open problem: credit assignment over a silent-to-noisy
transition, or a training horizon question, or both.

## Task 4 follow-up — re-tested R-STDP now that the network conducts

Old result (silent network): eligibility traces + signed dopamine + shaping +
explore, best case 1.25 → 2.38 food, inside noise, reflexes present throughout.

Re-ran the same overlays on the now-conducting network, 6–8 seeds, T=5000–6000:

**Reflex intact (sanity check — does R-STDP still help/hurt when the reflex is
already doing the steering):**

| condition | food |
|---|---|
| baseline (frozen reflex) | 20.75 ± 3.24 |
| + eligibility | 17.38 ± 3.82 |
| + eligibility + signed | 7.88 ± 2.81 |
| + elig + signed + shaping | 18.88 ± 3.30 |
| + elig + signed + shape + explore | 20.38 ± 3.31 |

No overlay beats plain baseline; signed dopamine alone hurts. Consistent with
reflex already owning the steering — there's no gap left for these to fill.

**Reflex deleted — the actual test of whether R-STDP now has signal to work
with, since sensor→motor pool spiking finally correlates:**

| condition | food |
|---|---|
| baseline plasticity, no reflex | 3.25 ± 0.58 |
| + eligibility, no reflex | 5.12 ± 1.37 |
| + eligibility + signed, no reflex | 5.00 ± 1.43 |
| + elig + signed + shaping, no reflex | 5.00 ± 0.95 |
| + elig + signed + shape + explore, no reflex | 4.88 ± 1.30 |
| reflex present but unfrozen (plastic) | 16.17 ± 3.26 |
| reflex present, unfrozen + eligibility | 14.17 ± 3.93 |

Eligibility traces now do *something* — 3.25 → ~5.0, a real if small lift,
consistent with the earlier finding that they were reasoning-correct but had
no signal to credit on the old silent network. But it's nowhere near closing
the gap to reflex-present performance (20.75) or the oracle ceiling (26–27).
Unfreezing the reflex synapses (rather than deleting them) keeps most of the
performance (16.17) — STDP reshaping them doesn't destroy them, unlike the old
silent network where unfreezing made things strictly worse.

**Reading:** conduction was necessary but not sufficient. A single ~6000-step
lifetime of vanilla reward-modulated STDP, even with eligibility traces, does
not discover a sensor→motor mapping from a cold, K-nearest-random recurrent
topology. Candidates for next steps, not attempted here: longer training
horizons, evolutionary/breedstock selection over the STDP-plastic weights
(the repo already has `jules/run_breedstock.py` for this), a denser/more
structured initial connectivity between sensory and motor pools before
learning takes over, or accepting the reflex arcs as a fixed "spinal" layer
and scoping the emergence claim to what's built on top of them.

## Task 4, round 2 — dense sensor↔motor wiring + evolutionary selection

R-STDP within one lifetime tops out around food≈5 with reflexes deleted. Two
remaining candidates from the list above, combined: (b) breedstock/evolutionary
selection over the plastic weights, (c) denser initial sensor↔motor
connectivity, so there's a structural pathway for selection to shape.

**What was built** (`jules/organoid_simulator/breedstock_dense_noreflex.py`):
reflex arcs still deleted; 4 new random sensor-pool→motor-pool synapses added
per motor neuron (280 total, small random excitatory init — Dale-respecting,
nothing pre-tuned toward chemotaxis) on top of the existing K-NN topology;
these plus every other non-reflex synapse become the GA genome. STDP off
during evaluation (evolution is the only optimizer, per the existing
`creature_breedstock.py` design and its own finding that STDP had ~no
behavioral effect on the pre-conduction-fix network). Fitness = reward-event
count (`food_eaten`) directly, no survival blend, matching the ask.

**Scale (reduced pilot, not the full plan):** pop 16, 18 generations, 5 fixed
train seeds at T=3000 (~14 min total). The repo's own `BREEDSTOCK_PLAN.md`
calls for pop 48 / 15 seeds — this is a cheaper proof-of-concept, not that.

| generation | best | pop mean |
|---|---|---|
| 0 | 4.40 | 3.17 |
| 3 | 6.00 | 4.24 |
| 10 | 7.40 | 5.15 |
| 17 (final) | 8.40 | 5.80 |

Champion held out on 8 fresh seeds, T=6000: **7.88 ± 1.66** food.

| condition | food |
|---|---|
| no reflex, untrained (random weights) | 3.25 ± 0.58 |
| no reflex, best R-STDP lifetime learning | ~5.0 |
| no reflex, dense wiring + evolution (this) | **7.88 ± 1.66** |
| reflex intact (frozen) | 20.75 ± 3.24 |
| oracle ceiling | ~26.8 |

**Reading:** the combination clears the R-STDP ceiling — real signal, not
noise (still climbing at generation 17, not yet plateaued: pop mean fitness
still trending up in the last few generations, so more generations/population
would very likely go higher than 7.88). It confirms selection *can* build
partial sensorimotor competence from a structurally-scaffolded-but-functionally-blank
network. It does not close the gap to the reflex-intact brain (~38% of it) or
the oracle (~29%) at this pilot's scale. Whether a full-scale run (pop 48,
more generations, more seeds) closes the rest of the gap, or whether the
reflex arcs are doing something structurally irreplaceable at this network
size, is still open — worth running at full scale before concluding either
way.

One caveat worth being explicit about: adding fixed sensor→motor *synapse
slots* is a structural change, even though their function is entirely
selection-shaped, not hand-tuned. It's closer in spirit to real cortex having
anatomical sensorimotor tracts that experience then shapes than to scripting
behavior — but it's a judgment call, and readers should weigh it as such
against the project's "no hardwiring" constraint.

## Task 4, round 3 — structural plasticity (the real mechanism, not seeded slots)

The dense-wiring pilot above (round 2) hand-placed 280 sensor→motor synapse
*slots*, which is a structural judgment call against the no-hardwiring
constraint even though their function was evolved, not tuned. index.html
already has the actual mechanism that avoids that call entirely: pruning,
reward-triggered synapse growth (`triggerRewardGrowth`, correlation-scored,
fires on each food-eat), and morphogen/migration self-organization — gated
behind the `dev` toggle (default OFF). `creature_embodied.py` deliberately
excludes all three ("Topology is FIXED so the loop vectorizes"), so it had
never been tested across seeds — only ever judged from a single browser run,
which is exactly the failure mode the testing protocol exists to avoid.

**Feasibility check first (this was the actual question asked):** yes, no
GPU needed. It's plain typed-array/numpy math, O(N) or O(E) with small
constants — growth samples 15 candidates × 8–16 attempts per eat event,
pruning walks the synapse list once every 120 steps. Confirmed by porting
it: 3000 steps with growth+pruning+self-organization all on took ~3.8s in
pure Python; the browser's typed-array JS will be faster than that, not
slower. Performance was never the blocker.

**Built:** `creature_embodied_structural.py` — pre/post/weight/is_reflex and
per-synapse activity/age now grow (array concatenation, on each eat event)
and shrink (boolean-mask filter, every 120 steps) during a run, instead of a
fixed-shape topology. Reuses the conduction+motor-fixed constants unchanged.

**Result, single lifetime, reflex intact (sanity check):** structural
plasticity on (20.67 ± 3.88) vs. off (24.50 ± 2.17, fixed topology) — no
significant difference, as expected; reflex is already doing the steering,
nothing for growth/pruning to fix.

**Result, reflex deleted — the actual test:**

| condition | food | synapses grown | pruned |
|---|---|---|---|
| fixed topology (no structural), no reflex | 5.12 ± 0.78 | 0 | 0 |
| structural plasticity ON, no reflex | 3.62 ± 1.09 | 34.2 | 1.5 |

**No lift, possibly a slight cost.** One lifetime of correlation-triggered
growth adds ~34 new random-ish synapses (limited by how rarely the creature
eats when it can't steer — few eat events means few growth triggers) and
doesn't out-forage plain in-lifetime STDP on the fixed topology. This is a
different mechanism from the round-2 evolutionary result (7.88 ± 1.66) and,
unsurprisingly, doesn't reach it: growth+pruning here has no fitness signal
across lifetimes, no selection retains "useful" synapses that happened to
form — a single lifetime's random correlational growth is just STDP with
extra steps. **Structural plasticity is real, cheap, and correctly wired now
that the network conducts — but on its own, within one lifetime, it isn't
the answer.** The natural next test (not run): structural plasticity's
per-lifetime growth *combined* with the round-2 evolutionary loop — let each
generation's lifetime also grow/prune synapses, and let selection retain
genomes whose growth built something useful, rather than the round-2 pilot's
fixed synapse-slot scaffold. That would let selection act on real,
self-organized wiring instead of pre-placed slots, closing the one remaining
judgment-call gap from round 2.

## Task 4, round 4 — evolution + real structural growth, no pre-seeded slots

Combines round 2 (evolutionary selection) with round 3 (real growth/pruning)
the way round 3 flagged: genome = initial weights on the **generic** K-NN
topology (no dense sensor→motor slots this time, no hand placement at all),
reflex arcs deleted (inert, weight 0). Each genome's lifetime fitness eval
runs with structural plasticity ON (growth+pruning, self-organization off for
GA speed — see caveat below) plus ordinary STDP on top of the evolved starting
weights. Selection only ever sees "evolved starting wiring + what this one
life's own growth/pruning/STDP did with it." Fitness = reward-event count
(food_eaten), same as round 2.

Pilot scale (smaller than round 2 — structural sim is slower per step): pop
12, 14 generations, 4 train seeds, T=3000 (~18 min).

| generation | best | pop mean |
|---|---|---|
| 0 | 3.50 | 2.54 |
| 5 | 4.75 | 3.38 |
| 10 | 5.00 | 3.54 |
| 13 (final) | 5.50 | 3.65 |

Champion held out on 8 fresh seeds, T=6000: **7.12 ± 0.78** food.

| condition | food |
|---|---|
| no reflex, untrained | 3.25 ± 0.58 |
| no reflex, best R-STDP lifetime learning only | ~5.0 |
| no reflex, structural plasticity alone (round 3) | 3.62 ± 1.09 |
| no reflex, dense pre-seeded slots + evolution (round 2) | 7.88 ± 1.66 |
| **no reflex, generic topology + evolution + real structural growth (this)** | **7.12 ± 0.78** |
| reflex intact (ceiling reference) | 20.75 ± 3.24 |
| oracle | ~26.8 |

**Reading:** statistically indistinguishable from round 2 (7.12±0.78 vs.
7.88±1.66, overlapping) — but this time with no pre-placed sensor→motor
slots and a tighter standard error despite a smaller pilot. Evolution,
given nothing but a generic topology and the real growth/pruning mechanism
already in index.html, finds its own way to roughly the same partial
competence round 2 got by scaffolding. That closes the round-2 judgment-call
gap: nothing here was placed by hand, only selected. Still climbing at
generation 13 (best 5.0→5.5 in the last few gens, not plateaued) and still
well below the reflex-intact reference (~34%) and oracle (~27%) — a
full-scale run (bigger population, more generations, self-organization back
on) is the obvious next step if more of that gap is worth chasing.

**Caveat:** self-organization (position drift) was left off for this pilot to
keep per-genome evaluation fast enough to run at all in this session (~3x
speedup). Growth is still distance-scored, just against each neuron's fixed
initial 3D position rather than a position that drifts with activity over
development — a real simplification versus index.html, worth revisiting if
this direction gets scaled up.

## Files changed

- `index.html` — `SYN_GAIN`, `MOTOR_SPEED_SCALE` constants + 2 call sites.
- `jules/organoid_simulator/creature_embodied.py` — same fix, plus the
  reflex-clip fidelity bug (`exc_syn`/`inh_syn` now masked by `plastic`).
- `creature_embodied_rstdp.py`, `run_experiments.py` — harness, as supplied,
  defaults updated to the tuned constants.
- `sweep_conduction.py` — new, the gain/rate/σ/steering sweep tool used above.
- `jules/organoid_simulator/breedstock_dense_noreflex.py` — new, the (b)+(c)
  scaffold + GA runner. Reuses GA operators from `creature_breedstock.py`.
- `champion_dense_noreflex.npy`, `ga_dense_noreflex.log`,
  `ga_dense_noreflex_result.json` — round 2 pilot artifacts.
- `jules/organoid_simulator/creature_embodied_structural.py` — new, structural
  plasticity (pruning/growth/self-organization) ported to the headless harness.
- `run_ga_structural_noreflex.py`, `champion_structural_noreflex.npy`,
  `ga_structural_noreflex.log` — round 4 pilot artifacts.
- `jules/organoid_simulator/creature_embodied_combined.py` — new, layers
  R-STDP (eligibility/signed dopamine/explore/shaping) on top of the real
  structural-plasticity substrate.
- `run_ga_combined_noreflex.py`, `champion_combined_noreflex.npy`,
  `ga_combined_noreflex.log` — Task 5 combined pilot artifacts.

## Generalization track — pulling in `brain.js` / `evolution_arena.html`

Other-branches survey (all diverge from ~July 20–21, before the Aug conduction/
motor fixes, never merged into `main`):

- **`main.-Evo`** — `brain.js`: an instantiable `Brain` class (no globals),
  contract `new Brain(genome)` → `b.step([sensors]) → [mL, mR]`, `b.reward(x)`.
  Same biology as index.html, decoupled from any specific arena/body already —
  this is the brain/task interface discussed earlier, pre-built. Plus
  `evolution_arena.html`, a live population-of-brains UI. Pulled onto this
  branch (`brain.js`, `evolution_arena.html`).
- **`evolution-experiments-prototype`** — `evolution_experiments.py`: seven
  brain-combination/inheritance operators (structural combination,
  developmental inheritance, co-culture fusion, neural transplantation, trait
  crossover, experience transfer, multi-stage pipeline) — richer than this
  project's plain weight-vector GA. Not pulled in yet (see below).
- `creature-spiking-sweep`, `feature-realistic-spiking-organoid` — superseded;
  their content already exists in current `main`.

**`brain.js` had the identical silent-network bug.** Measured directly (Node,
`sweep_brainjs_conduction.js`): interneuron pool rate 0.00064/step — the same
noise-kick floor as the pre-fix 600-neuron network. `injMult` (22) and
`reflexScale` (3, boosted to 6 in `evolution_arena.html`'s `GENOME_TWEAKS`)
compensate by cranking sensor injection and reflex-arc weight, exactly the
workaround this project's diagnosis flagged as a dead end (steering r only
+0.14 at 160× reflex weight) — never fixing the recurrent core. Independent
confirmation of the same finding from a differently-authored implementation.
Also independent confirmation on the motor side: `evolution_arena.html`'s own
`SPEED_GAIN = 0.5` and `TURN_GAIN = 10` are already far above index.html's
original broken `0.007` — its author clearly hit the same "speed pinned at
floor" problem and worked around it empirically, without a name for what was
wrong.

**Fix ported:** added `synGain` to `Brain` (genome-overridable, mirrors
`SYN_GAIN`), scaling recurrent transmission only. Swept for this network's
size (N=120, K=6–10 — much smaller than the 600-neuron network, needed its own
calibration, not the same constant): interneuron rate 0.00064 → ~0.025–0.033
at synGain 60–80; picked **70**. Stable over 6000+ steps, no NaN/Inf
(`sweep_brainjs_conduction.js`). **Not yet behaviorally validated** — no
steering-correlation or foraging-score measurement on `brain.js` yet, same
protocol as the Python-side fix still needs to run here before trusting it.

**`evolution_experiments.py`'s results are confounded by the same bug.**
Its `results_evolution.json` shows `food_eaten` of 0–1 across all seven
experiments (Experiment A: 0, B: 1, C: 0, D: 0, E: 1, F: 0, G: 0) — the same
signature as this project's original diagnosis. The operators themselves are
well-designed and worth having, but none of their reported outcomes can be
trusted as "this inheritance mechanism works/doesn't" until re-run on a fixed
substrate. Not pulled in yet — recommend porting the operators onto
`creature_embodied.py` (which has the Aug fixes) or `brain.js` (once
validated) and re-running before drawing any conclusion about them.

**Status / next steps, not done in this pass:**
1. Steering-correlation + foraging-score validation of `brain.js` at
   synGain=70, matching the Python-side testing protocol (6–8 seeds).
2. `evolution_arena.html`'s `GENOME_TWEAKS` may need re-tuning now that the
   recurrent network actually conducts (injMult/reflexScale were compensating
   for silence that no longer needs compensating).
3. Re-run `evolution_experiments.py`'s seven operators on a fixed substrate
   before judging any of them.

## Generalization track, follow-up — brain.js validation: gate NOT passed

Ran the same protocol used on the 600-neuron network (steering correlation +
food vs. oracle, multiple seeds) against `brain.js`/`evolution_arena.html`'s
actual foraging loop (`test_brainjs_foraging.js`), not just the synthetic
random-sensor conduction probe from before.

| condition | food | steering r |
|---|---|---|
| oracle ceiling (creep speed 0.0018) | 25.3 | — |
| synGain≈0 (pre-fix, silent) | 0.50 ± 0.18 | 0.005 |
| synGain=70 (chosen default) | 0.88 ± 0.28 | 0.021 |
| synGain=150 | 1.67 ± 0.73 | — |
| several injMult/reflexScale/synGain rebalances tried | 0.17–2.83 | −0.044 to +0.031 |

**The gate is not clearly positive.** Unlike the 600-neuron Python network
(−0.017 → +0.175, a clean and repeatable jump), synGain barely moves steering
correlation here across every combination tried, and food stays noisy and low
against a 25.3 oracle ceiling. Per this project's own testing protocol
("do not proceed until the gate is clearly positive"), **`brain.js` should
not be treated as fixed** — raising `synGain` alone does not reproduce the
600-neuron result on this smaller (N=120), differently-parameterized network.

Diagnostic work not yet done, needed before touching this further: hop-by-hop
tracing (injection→sensor-pool spiking, sensor-pool→motor-pool spiking,
matching the original August diagnosis) specific to this topology; more
seeds (4–6 used here, noisy at this sample size); checking whether
`METAB_BASE=0.075` (2.5× the 600-neuron network's 0.030) is killing creatures
before conduction has time to matter (observed survival ~400–980 of a
3000–4000-tick cap); and whether N=120's much sparser interneuron pool
(~52 neurons) can support the same kind of avalanche dynamics at all, or
needs its own K/connectivity retuning, not just a gain constant.

**Status:** `brain.js` is pulled in and has a `synGain` knob, but is
NOT validated — don't build on it (evolution_experiments.py port, arena
re-tuning) until this gate passes. The 600-neuron `creature_embodied.py` /
`index.html` substrate remains the only one with a confirmed, gated fix.

## Generalization track, follow-up 2 — homeostatic self-tuning gain: fixes calibration, not steering

Implemented the Nature Comms-style fix: `synGain` is now a genome-overridable
*base* (default 1.0, neutral) multiplied by `adaptiveGain`, a self-tuning
scalar that starts conservative and adjusts itself every 20 ticks (±5% step,
clamped [0.5, 2000]) toward whatever value keeps the recurrent population
(motor pools + interneurons, excluding externally-injected sensory pools)
near `genome.homeoTarget` — no more hand-picked constant to sweep per network.

**The self-tuning mechanism works as designed, verified directly:** over
6000 ticks, `adaptiveGain` converged to ≈153 and held the recurrent
population's firing rate at 0.031 against a target of 0.026 — close, stable,
no NaN/Inf, no per-network sweep required.

**It did not close the gate.** Also fixed a real bug found in the process
(`this.reflexScale = this.genome.reflexScale || 3.0` treats an intentional
`reflexScale: 0` override as falsy and silently falls back to 3.0 — same for
`injMult`; changed `||` to `??`). With that fixed, reflex-deleted now
correctly differs from reflex-intact:

| condition | food | steering r |
|---|---|---|
| oracle ceiling | 25.3 | — |
| reflex intact, self-tuning gain | 0.63 ± 0.25 | 0.049 |
| reflex deleted, self-tuning gain | 0.50 ± 0.18 | −0.004 |

Reflex still contributes the only nonzero signal, and it's small — nothing
like the 600-neuron network's reflex-intact +0.175. Activity now reaches a
sensible, self-calibrated level (confirmed), but that activity still isn't
carrying steering information from sensor pools to motor pools. Necessary,
not sufficient — same conclusion this project already reached on the
600-neuron network's *no-reflex* condition, but here it holds even with
reflex intact, which is new and worse.

**Conclusion: this isn't a gain problem anymore, on either fixed or
self-tuning terms.** Something structural to `brain.js`'s N=120
parameterization — proportional pool sizing (sp≈7, mp≈13 vs. the 600-neuron
network's fixed 20/35), the stride-based reflex wiring formula, the shorter
distance-decay scale, or the different STDP dopamine formula (`0.03 +
dopamine*5` vs. the Python side's `0.1 + dopamine*4`) — is not carrying
sensor-to-motor correlation the way the larger network does. That needs the
same kind of hop-by-hop tracing (injection→sensor-pool spiking,
sensor-pool→motor-pool spiking) that solved the original diagnosis, not
another gain sweep. Recommend treating that as its own diagnostic pass rather
than continuing to guess at constants — three different gain strategies
(fixed 70, rebalanced fixed, self-tuning) have now all landed in the same
near-zero place.

**Status, unchanged:** `brain.js` remains unvalidated for behavioral use.
The 600-neuron `creature_embodied.py`/`index.html` substrate is still the
only one with a confirmed, gated fix.

## Task 5 — combining mechanisms: evolution + structural plasticity + R-STDP together

Every mechanism tested alone on the no-reflex 600-neuron network gave a
small or zero lift over the no-reflex floor (3.25 ± 0.58 food):

| condition (no reflex, plain unless noted) | food |
|---|---|
| R-STDP alone (elig+signed+shape+explore, fixed topology) | ~5.0 – 5.12 |
| structural plasticity alone (growth/prune, plain STDP) | 3.62 ± 1.09 |
| dense pre-seeded sensor↔motor slots + evolution | 7.88 ± 1.66 |
| generic topology + real structural growth + evolution | 7.12 ± 0.78 |

Question asked: does combining them stack? Built
`jules/organoid_simulator/creature_embodied_combined.py` (real structural
growth/prune + full R-STDP overlay, both layered on the already-fixed
conduction/motor-gearing constants) and evolved initial weights on top of
it — `run_ga_combined_noreflex.py`, same generic no-reflex K-NN topology
and GA settings as the structural+evolution round (round 4), with R-STDP
turned on during every genome's lifetime (`elig=True, signed=True,
explore=1.0, shape_gain=0.55`).

Pilot scale: pop 12, 14 generations, 4 train seeds at T=3000; champion
re-evaluated on 8 fresh held-out seeds at T=6000 (same protocol as round 4):

| generation | best | pop mean |
|---|---|---|
| 0 | 4.75 | 3.21 |
| 3 | 7.25 | 4.33 |
| 13 (final) | 7.25 | 4.40 |

Champion held out on 8 fresh seeds, T=6000: **8.00 ± 0.98** food
(per-seed: 2, 10, 11, 7, 9, 7, 7, 11).

| condition | food |
|---|---|
| **evolution + structural + R-STDP (combined, this)** | **8.00 ± 0.98** |
| evolution + structural alone (round 4, plain STDP) | 7.12 ± 0.78 |
| dense slots + evolution alone (round 2, plain STDP) | 7.88 ± 1.66 |
| reflex-intact ceiling (reference only) | 20.75 ± 3.24 |
| oracle ceiling (reference only) | ~26–27 |

**Combining did not stack.** 8.00 ± 0.98 is statistically indistinguishable
from both prior evolution rounds — the error bars overlap heavily with
7.12 and 7.88. R-STDP's eligibility/dopamine machinery adds essentially
nothing on top of what evolution + structural growth already found, the
same small-to-nil effect it had when tested alone against the fixed
topology (~5.0 vs. that condition's own no-reflex floor of 3.25).

**Reading:** evolution is doing effectively all of the no-reflex lift in
every combination tried so far; neither within-lifetime STDP/R-STDP nor
structural growth adds to it, alone or stacked together. That's a
meaningful negative result — it points at the bottleneck being something
evolution reaches (the initial recurrent wiring) that none of the
lifetime mechanisms currently reach on their own, rather than "which
plasticity rule is missing." Everything tried in the no-reflex condition
still tops out around 7–8 food, roughly a third of the reflex-intact
ceiling (20.75) and under a third of the oracle (~26–27) — evolved,
emergent sensorimotor competence is real here but still clearly bounded
well below what the hand-wired reflex or perfect chemotaxis reach.

**Status:** `champion_combined_noreflex.npy` and `ga_combined_noreflex.log`
hold the run artifacts. This closes out the "combine the weak mechanisms"
question the way the testing protocol requires — honestly, against the
oracle and reflex-intact references, not as a declared win.

## Task 6 — GPU port: batched evolution, and did the climbing curves have a real cap?

Built `gpu_evolve.py`: the whole population x seed batch runs as one set of
vectorized PyTorch tensor ops per timestep (shared topology, only the plastic
weight tensor differs per batch row — `(B, n_plastic)`), instead of a serial
Python loop per genome. Installed Python + CUDA PyTorch on the user's RTX
5070 via the device bridge, validated the port against the CPU reference
(steering r +0.21 mean, reflex intact, matching the CPU gate's sign and
magnitude; no NaN/instability) before trusting any result from it. Documented
simplification: fixed topology only (no structural growth — hard to batch,
variable shape per genome) and no R-STDP overlay by default (Task 5 found it
added ~nothing; kept as an optional `--learning surprise` mode instead, see
below). Same conduction/motor fixes, same Dale/STDP/homeostasis biology.

First GPU run (pop 48, 50 gens, reflex curriculum 1.0->0.0 over 35 gens):
held-out 4.62 ± 1.44 food, steering r 0.009 — no better than the CPU pilots,
ruling out "just needed a cold-start ramp" as the missing piece.

**Question asked next: were the CPU pilots' climbing curves (round 2: still
rising at gen 17/18; round 4: still rising at gen 13) actually capped, or did
they just run out of budget?** Re-ran round 2's exact condition (dense
sensor->motor wiring, `--dense-k 4`, plain STDP, no reflex from generation 0,
no curriculum) at real scale: pop 48, **100 generations** (vs. the pilot's
18), same 4 fixed training seeds, T=3000.

| metric | CPU pilot (pop 16, 18 gens) | GPU (pop 48, 100 gens) |
|---|---|---|
| best training fitness | 8.40 | **14.75** |
| held-out food (8 fresh seeds, T=6000) | 7.88 ± 1.66 | **6.75 ± 2.27** |
| held-out steering r | (not measured) | **-0.006** |

**Training fitness really did keep climbing — nearly 2x higher with 5.5x the
compute.** But held-out performance did not follow it up; if anything it's
nominally lower, and heavily overlapping in error with the pilot. Steering
correlation on the held-out champion is flat zero, same as every no-reflex
condition ever tested.

**Reading: this is overfitting to the 4 training seeds, not undiscovered
capacity.** 100 generations of selection pressure against a fixed, small
set of food layouts found genomes that exploit quirks of those specific
seeds (memorized food/noise sequences) without learning anything that
transfers to fresh arenas. This explains why the pilots' curves looked
promising — the climb was real, it just wasn't climbing toward competence.
More generations on a fixed small seed set was the wrong lever; more/varied
training seeds per generation (at real compute cost, since fitness eval
scales with `pop x seeds`) is the actual fix implied by this result, and
wasn't tested here.

## Task 6, continued — three-factor "surprise" learning (eligibility x reward-prediction-error)

Requested angle: credit assignment should tag the neurons that were active
*on the way to* the food, not just the ones firing in the exact instant of
eating, and should be driven by whether reward was *surprising* (a
predictive/value-error signal), not by raw reward magnitude. The existing
STDP rule does neither: it updates weight only at the instant pre/post fire,
gated by the *current* dopamine level, with a trace horizon of ~25 steps
(`TRACE_DECAY=0.96`) — closer to "whatever was active near this specific eat
event" than "what led up to it."

Added `--learning surprise` to `gpu_evolve.py`: a slow eligibility trace
(`elig`, decay 0.995, ~200-step horizon) accumulates Hebbian pre/post
coincidences continuously, independent of reward — this is what "remembers
the neurons that led to the food." A running value estimate `V` tracks
expected dopamine (leaky average); actual weight change every step is
`elig x (dopamine - V)` — proportional to reward-*prediction-error*
(Schultz-style surprise), not raw dopamine, and applied through the
eligibility trace so credit reaches back through the approach rather than
gating on the eat-instant alone.

Validated the mechanism alone first (reflex intact, sanity check, no
evolution): stable, no NaN, steering r stayed positive (+0.10, lower than
plain STDP's +0.18 at this short T=2000 but clearly in the same functioning
regime) — confirms the new rule doesn't break the closed loop.

Full run: pop 48, 100 generations, no reflex from gen 0, no dense wiring
(isolating the learning-rule change), same seed/T budget as Task 6's first
run.

| metric | dense+evolution, plain STDP (this task, first run) | evolution + surprise learning |
|---|---|---|
| best training fitness | 14.75 | 13.00 |
| held-out food (8 fresh seeds, T=6000) | 6.75 ± 2.27 | **9.25 ± 2.52** |
| held-out steering r, per-seed | -0.074, -0.028, 0.094, -0.033, -0.077, 0.061, 0.024, -0.015 | **0.068, 0.050, 0.054, 0.048, 0.075, 0.080, 0.048, 0.077** |
| held-out steering r, mean | -0.006 | **0.063** |
| held-out steering r, sign consistency | scattered (4 pos / 4 neg) | **8/8 positive** |

**This is the first no-reflex result in the whole project where steering
correlation is consistently positive across every held-out seed**, not
scattered around zero the way plain STDP, R-STDP, structural plasticity,
dense wiring, curriculum, and combined runs all were. Tight spread (SEM
0.005) across seeds is the actual signature of a real signal — a condition
that's truly at zero should flip sign roughly half the time across 8
independent seeds; this didn't flip once.

Reading, carefully: 0.063 is still well below the reflex-intact ceiling's
+0.175 — this is not competence on par with the hardwired reflex, and food
(9.25 ± 2.52) is noisy with two zero-food outlier seeds. But qualitatively
this is different from every prior negative result: crediting synapses by
*eligibility x reward-prediction-error* instead of *instantaneous STDP x raw
reward* is the first change in this whole project that produced a
directionally consistent no-reflex steering signal rather than noise. The
mechanism the user asked for — remembering the neurons that led to the food,
not just the ones firing at the reward instant, and gating on surprise
rather than raw reward magnitude — appears to be doing real work here.

**Not yet verified enough to call solved.** Before trusting this: rerun on
a different scaffold seed (rule out this specific topology being a fluke),
more held-out seeds (8 is thin for a claim this significant), and a direct
head-to-head against plain STDP on the *identical* topology/genome-init
conditions (this run had no dense wiring, the STDP comparison run did —
not a clean A/B yet). That's the next step, not a victory lap.

**Status:** `champion_surprise_scaled.npy`, `run_surprise.log`,
`champion_dense_stdp_scaled.npy`, `run_dense_stdp.log` hold both runs'
artifacts.

## Task 6, continued — replication check: didn't hold up

Per the write-up above, ran the same surprise-learning condition on a
different topology (`--scaffold-seed 7` instead of 42) before trusting the
first result — pop 32, 60 gens (lighter than the first run, enough to check
direction, not to push a new champion).

| metric | scaffold seed 42 (first run) | scaffold seed 7 (replication) |
|---|---|---|
| held-out food | 9.25 ± 2.52 | 8.88 ± 2.06 |
| held-out steering r, per-seed | all 8 positive, 0.048-0.080 | 0.031, 0.004, 0.001, 0.009, 0.026, -0.022, 0.023, -0.010 |
| held-out steering r, mean | 0.063 | **0.008** |
| sign consistency | 8/8 positive | 6/8 positive, 2 negative |

**Did not replicate.** Food stayed comparably high, but the steering
correlation collapsed to noise-level (0.008, two sign flips) on a different
network topology. The clean, all-positive signal from the first surprise run
was topology-specific, not a property of the eligibility x
surprise learning rule in general — the same conclusion the project has now
reached for every other single-run "this looks like it worked" result, which
is exactly why the testing protocol requires a replication before trusting
one.

**Corrected reading:** the surprise/eligibility mechanism is not obviously
better than plain STDP at solving no-reflex closed-loop steering. It may
still be doing *something* real on some topologies (worth more seeds/reps to
characterize, not investigated further here) and food output is at least as
good as every other condition tried, but the steering-correlation claim from
the first run does not survive replication and should not be treated as a
solved gate. Filed as a negative-with-an-asterisk result, not a positive one.

**Status:** all no-reflex conditions tested this session — GPU curriculum,
dense+evolution at scale, surprise learning (both topologies) — land in the
same 5-9 food / near-zero-to-noisy steering band the CPU pilots did. No
mechanism tried so far reliably produces the closed-loop steering signal the
reflex arcs provide for free. `champion_surprise_seed7.npy`,
`run_surprise_seed7.log` hold this run's artifacts.

## Task 7 — "hunting bootcamp": easy-mode pretraining before the real test

User's ask: teach and reinforce the hunting skill specifically, rather than
relying on rare full-difficulty reward events to drive learning. Added a
two-phase lifetime to `gpu_evolve.py` (`--bootcamp`): each genome first runs
an easy phase (bigger eat radius 0.006 vs. the real 0.0009, food always
respawns within 0.15 of the creature instead of uniformly over the arena,
metabolism cut to 0.4x, motor exploration noise annealed 0.35->0.05 across
the phase) with learning ON, so reward events happen constantly instead of
rarely and the credit-assignment machinery gets dense signal to work with.
Only the shaped weights carry into the real (scored, normal-difficulty)
phase — bootcamp food doesn't count toward fitness.

Paired with the surprise/eligibility learning rule (the one mechanism from
Task 6 that showed a real, if unreplicated, effect). Pop 48, 60 generations,
no reflex from gen 0, boot_t=1500 then train_t=3000, held out on 8 fresh
seeds at T=6000 (bootcamp phase included before the held-out score too, same
as training).

| condition | held-out food | steering r, mean | steering r sign consistency |
|---|---|---|---|
| surprise, no bootcamp (Task 6, seed 42) | 9.25 ± 2.52 | 0.063 | 8/8 positive |
| surprise, no bootcamp (Task 6, seed 7 replication) | 8.88 ± 2.06 | 0.008 | 6/8 positive, noise |
| **surprise + bootcamp (this task)** | **6.38 ± 0.43** | **0.028** | **7/8 positive** |

**Bootcamp didn't raise the ceiling — it collapsed the variance.** Food's
mean is actually the lowest of the three, but its standard error (0.43) is
4-6x tighter than every other no-reflex run this project has produced, and
every seed landed in a narrow 5-8 band with no zero-food failures. Steering
r sits between the original surprise result and its failed replication:
weaker than the lucky run, more consistent than the noise-level one.

**Reading:** this is a real result, just not the one being chased. Reward
density during training changed reliability, not competence. That's
consistent with what bootcamp actually changes (how much learning signal
the credit-assignment machinery gets per unit of simulated time) versus what
it doesn't change (the fundamental question of whether this network
architecture can represent and discover closed-loop steering at all). If the
ceiling itself is the target, bootcamp alone isn't it — but a *reliable*
mid-single-digit performer is a meaningfully different, more useful result
than a lottery between 0 and 15 depending on seed, and it's the only
no-reflex condition this project has produced where every seed did roughly
the same thing.

**Status:** `champion_bootcamp_surprise.npy`, `run_bootcamp_surprise.log`
hold the run artifacts. Untested: bootcamp + plain STDP (isolate whether the
reliability gain is from bootcamp itself or specific to the surprise rule),
and bootcamp at larger scale now that per-genome variance is low enough
that more seeds might resolve real differences instead of just noise.

## Task 8 — "fixed retina" eye pool: higher sensory resolution didn't help

Dave's proposal: keep the same "learns on its own, no hardcoding" constraint
but ask whether the *sensing* itself is the bottleneck — the existing 2-lobe
smell antennae blend every food item's direction into one ambiguous number
per side, which may be too degenerate a signal for STDP/evolution to ever
recover a clean sensor->motor mapping from. The fix explored: a fixed,
unlearned "retina" (biologically honest — real retinas do fixed
preprocessing; only cortex is plastic) of 20 directionally-tuned
photoreceptors across a 150-degree forward field of view, using the exact
same distance/cosine-lobe physics the smell antennae already use, just
resolved into many narrow lobes instead of two wide ones. No object
detection, no segmentation, no food identity — still a raw gradient field,
just spatially richer. Added as `POOLS['eye']` (carved from the interneuron
budget, N stays 600), injecting into its own pool exactly like every other
sensor, with a matching steering-correlation diagnostic (`dEye` vs. `dM`,
parallel to the existing smell diagnostic).

Ran the identical config to Task 7's best result (surprise learning +
bootcamp, pop 48, 60 gens, no reflex) with the eye pool active alongside the
existing smell sensors, for a clean before/after:

| condition | held-out food | steering r (smell) | steering r (eye) |
|---|---|---|---|
| bootcamp + surprise, no eye (Task 7) | 6.38 ± 0.43 | 0.028 (7/8 positive) | n/a |
| bootcamp + surprise, **with eye** (this task) | 6.50 ± 1.21 | 0.024 (6/8 positive) | **-0.027 (2/8 positive)** |

**The eye pool did not help, and the extra sensory resolution came with a
cost.** Food mean is essentially unchanged, but its SEM nearly tripled
(1.21 vs. 0.43) and per-seed spread widened back out (1-12 vs. the tight 5-8
band Task 7 achieved) — the reliability gain bootcamp bought got partly
undone. Steering correlation through the eye channel itself is noise-level
at best, arguably slightly negative (2/8 positive, mean -0.027) — worse than
the existing smell channel, which stayed roughly where it was.

**Reading: sensory resolution was not the bottleneck, and more input
channels without a way to route them made things marginally worse.** 20 new
photoreceptor channels are 20 new things the network has to structure
connections around with the same fixed training budget (same seeds, same
generations); nothing in the K-NN topology-building process routes those
new channels toward the motor pools any better than chance, so on average
they likely diluted the connectivity/plasticity budget rather than adding
usable signal. This doesn't rule out vision as a concept — it rules out
*just adding more raw channels* as the fix. A retina that mattered would
probably need either a genuinely topographic/patterned connection structure
to the eye pool (mimicking how a real optic nerve maps retina position to
cortical position, rather than the same random-distance-based wiring every
other sensor gets) or a training budget scaled to the added input
dimensionality — neither tried here.

**Status:** `champion_bootcamp_surprise_eye.npy`, `run_bootcamp_surprise_eye.log`
hold the run artifacts. Recommend not pursuing the eye pool further without
a specific hypothesis for how it would connect *differently* than existing
sensors, since simply adding it did not clear the same bar dense wiring,
structural growth, curriculum, or scale all failed to clear either.

## Task 9 — Free Energy Principle (predictability) learning: best result yet, and it replicates

Motivated by reading the bl1 repo (DishBrain/Kagan et al. 2022 replication):
biological/simulated cortex there isn't taught with a reward scalar at all.
Plasticity is plain, always-on Hebbian STDP — no dopamine gating, no
eligibility-trace-times-TD-error. "Reward" is implemented entirely on the
*environment* side: success makes incoming stimulation more predictable and
coherent; failure makes it chaotic and unpredictable. The network's own
ordinary Hebbian machinery naturally strengthens whatever structure the
predictable input has, and naturally can't build structure out of noise.
No new weight-update rule was needed — the opposite of every other
mechanism tried in this project, all of which added machinery to the
learning rule itself (eligibility traces, surprise-gating, structural
plasticity, curricula). This one removes machinery and moves the "credit
assignment" outside the brain entirely.

Implementation, two independent pieces:
- `--learning fep`: plain unmodulated Hebbian STDP (`A_PLUS`/`A_MINUS`, same
  constants as always) with **no dopamine multiplier at all** — contrast
  with `stdp` mode's always-nonzero `0.1 + dopamine*4` gate.
- `--fep-punish`: environment-side punishment. A per-creature timer fires a
  burst of pure random noise (replacing the real smell L/R signal, and eye
  signal if `--use-eye` is on) for `--fep-punish-t` steps (150) whenever the
  creature gets too close to a wall (`--fep-wall-thresh`, 0.7) or goes too
  long without eating (`--fep-timeout-steps`, 800) — i.e., failure states.
  Interoceptive channels (hunger, tiredness, food-proximity) are left alone;
  only the two channels a real DishBrain-style predictability signal would
  plausibly act on get scrambled.

Ran at the established real-scale config (pop 48, 60 gens, no reflex from
gen 0, held out on 8 fresh seeds, T=6000) on the confirmed scaffold-seed 42
topology, then — per this project's replication protocol (surprise learning
looked great on seed 42 and failed on seed 7) — reran the identical
config on scaffold-seed 7 before trusting it:

| condition | held-out food | steering r (smell), mean | positive seeds |
|---|---|---|---|
| R-STDP alone (fixed topology, no growth) | ~5.0-5.12 | n/a | n/a |
| structural alone (plain STDP, no evolution) | 3.62 ± 1.09 | n/a | n/a |
| dense + evolution (100 gens, overfit) | 6.75 ± 2.27 | -0.006 | — |
| surprise learning, seed 42 | 9.25 ± 2.52 | 0.063 | 8/8 |
| surprise learning, seed 7 (replication) | 8.88 ± 2.06 | 0.008 | 6/8 |
| bootcamp + surprise | 6.38 ± 0.43 | 0.028 | 7/8 |
| **FEP, seed 42 (this task)** | **7.75 ± 2.07** | **0.080** | **6/8** |
| **FEP, seed 7 (replication)** | **10.62 ± 2.08** | **0.041** | **5/8** |
| reflex-intact ceiling (reference) | 20.75 ± 3.24 | n/a | n/a |
| oracle ceiling (reference) | ~26-27 | n/a | n/a |

**This is the first mechanism whose no-reflex closed-loop steering signal
held up, in the same direction, on both topologies tested.** Surprise
learning's strong seed-42 result (0.063, 8/8 positive) essentially
vanished on seed 7 (0.008, barely above noise) — a genuine failure to
replicate. FEP's steering correlation is smaller on seed 42 (0.080 — even
higher than surprise's 0.063) and drops on seed 7 (0.041) but **stays
clearly positive on both**, and food count is the highest of any honest
(non-overfit) no-reflex condition on record: 7.75 on seed 42, 10.62 on
seed 7 — both above every prior no-reflex mechanism's mean.

Caveats, stated plainly: per-seed spread is still wide (1-18 food on seed
42, 3-19 on seed 7), positive-seed fraction is 6/8 and 5/8 — not the clean
8/8 surprise learning showed on its one good topology, and not proof this
scales past two topologies. But two different topologies pointing the same
direction, with no new weight-update machinery, is the strongest evidence
of a general (not topology-specific) effect this project has produced.

**Status:** `champion_fep.npy`/`run_fep.log` (seed 42),
`champion_fep_seed7.npy`/`run_fep_seed7.log` (seed 7, replication) hold the
run artifacts. Recommended next steps: a third topology to raise confidence
further; sweep `--fep-wall-thresh`/`--fep-timeout-steps` (untuned, first
guess values); try `--fep-punish` combined with `--use-eye` now that a
routing hypothesis exists (predictable vs. scrambled signal, not just more
channels) — Task 8's null result on the eye pool used reward-gated learning,
not FEP, and may not generalize.

## Task 10 — "best of both": surprise + FEP-punish does NOT stack

Natural next question: FEP's two pieces (plain Hebbian weight rule,
environment-side punishment) are independent flags — `--fep-punish` doesn't
require `--learning fep`. Tried combining FEP's environment punishment with
the *surprise* eligibility x TD-error weight rule instead of plain Hebbian,
reasoning that surprise's longer credit-assignment horizon might make even
better use of a cleaner (punishment-scrambled-on-failure) signal. Two
variants, both seed 42, both `--learning surprise --fep-punish`:

| condition | held-out food | steering r (smell), mean | positive seeds |
|---|---|---|---|
| FEP alone (Task 9, seed 42) | 7.75 ± 2.07 | 0.080 | 6/8 |
| surprise alone (seed 42) | 9.25 ± 2.52 | 0.063 | 8/8 |
| **surprise + fep-punish, no bootcamp** | 7.00 | **-0.031** | 3/8 |
| **surprise + fep-punish + bootcamp** | 6.25 ± 1.09 | **0.002** | 4/8 |

**Combining them made steering worse, not better, on both variants.**
Without bootcamp the mean correlation actually flips negative (-0.031,
only 3/8 positive) — worse than either parent mechanism alone. With
bootcamp added it's essentially noise (0.002, 4/8 positive) — the
tightest food-count SEM of the two combo variants (1.09) but no real
steering signal underneath it.

**Reading:** the two mechanisms aren't additive, and may actively interfere.
Best guess: surprise's value-tracking (`V` chasing dopamine, delta = dopamine
- V) is tuned around a *continuous, gradually-changing* reward signal;
FEP's punishment mechanic periodically saturates the sensory channels with
noise, which likely also corrupts the eligibility trace's Hebbian
coincidences during punished windows (there's nothing coherent for pre/post
firing to correlate with) right as the surprise rule is trying to assign
credit through that same trace. FEP's plain-Hebbian rule has no such
value-tracking to disrupt, and no eligibility trace to corrupt — the reason
it likely tolerates its own punishment mechanic while surprise's rule doesn't.

**Status:** `champion_combo_nb.npy`/`run_combo_nb.log` (no bootcamp),
`champion_combo_boot.npy`/`run_combo_boot.log` (with bootcamp) hold the run
artifacts. Recommend NOT combining these two mechanisms; FEP alone (Task 9)
remains the best-validated result. If revisiting: try `--fep-punish` with a
much shorter `--fep-punish-t` (currently 150 steps, ~1/4 of surprise's own
~200-step eligibility horizon) so punishment bursts don't wipe out an entire
eligibility window at a time.

## Task 11 — FEP punishment-parameter sweep: relaxing punishment trades steering for raw food

Task 9's `--fep-punish` defaults (punish_t=150, wall_thresh=0.7,
timeout_steps=800) were first guesses, never tuned. Swept three one-at-a-time
relaxations against that baseline, all seed 42, same pop/gens/heldout setup:

| variant | held-out food | steering r (smell), mean | positive seeds |
|---|---|---|---|
| **baseline (Task 9)** | 7.75 ± 2.07 | **0.080** | 6/8 |
| shorter burst (`--fep-punish-t 60`) | 10.75 | 0.016 | 5/8 |
| rarer starvation trigger (`--fep-timeout-steps 1500`) | 9.50 | 0.014 | 5/8 |
| looser wall trigger (`--fep-wall-thresh 0.85`) | 5.38 | 0.008 | 4/8 |

**Every relaxation increased food-count noise or dropped it, while
consistently gutting steering correlation** — the baseline's 0.080 is 5-10x
every relaxed variant's. Two of three variants (shorter burst, rarer
timeout) traded steering for a higher food mean; the third (looser wall)
lost on both.

**Why food and steering can diverge (raised by Dave, worth stating
explicitly): they measure different things.** Steering correlation is a
strict linear Pearson r between sensor L-R asymmetry and motor L-R
asymmetry — it isolates fine, moment-to-moment directional response to the
smell gradient specifically. Food count rewards ANY path to a food item,
including non-directional strategies: modulating forward speed off
food-close/hunger rather than turning, more total movement/exploration
(a bigger random walk collides with more food by luck alone), or a
nonlinear/threshold turning response a linear r doesn't capture. Less
punishment exposure appears to let evolution settle into a "wander more,
eat what you bump into" strategy that pads food count without building
real gradient-following — the "lucky wanderer" failure mode this project
has flagged before as categorically different from (and less interesting
than) genuine emergent steering, even when the food number looks better.

**Reading: punishment INTENSITY, not just its presence, is doing the work.**
All three relaxations reduce how often/how long the network's sensory input
actually gets scrambled; steering quality tracks that exposure almost
monotonically. This suggests the next experiment is the untested opposite
direction — MORE punishment, not less (shorter timeout, lower wall
threshold, longer burst) — rather than further relaxation.

**Status:** `champion_fep_shortburst.npy`/`run_fep_shortburst.log`,
`champion_fep_longtimeout.npy`/`run_fep_longtimeout.log`,
`champion_fep_tightwall.npy`/`run_fep_tightwall.log` hold the run artifacts.
Baseline params (Task 9's defaults) remain the best steering result on
record; recommend a harsher-punishment sweep next, evaluated on steering
correlation as the primary metric (not food count alone, given the
divergence found here).
