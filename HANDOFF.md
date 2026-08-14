# DigitalOrganoid — session handoff (Aug 14 2026)

## Where things stand right now
Candidate #3 (critical-period gating + event-gated eligibility +
punishment-timing gate) is fully closed out as of this handoff — none of its
three sub-tests beat the Task 9 FEP baseline. That exhausts
`CANDIDATE_MECHANISMS.md` shortlist items 0-3. RESULTS.md and this file are
up to date and committed locally on the PC (commit `0086fe8`), **not yet
pushed to GitHub** — the cloud session can't push (403, repo not in its
authorized set); push from the PC checkout when convenient.

**Candidate #4 (evolve-the-plasticity-rule) has NOT been started.** Per
Dave's "do 3 then 4," it's next up, but no code has been written for it yet
— paused before implementation began. See "Immediate next step" below for
what it involves.

## Project
"Digital life simulator": Izhikevich SNN brain (Dale's principle, STDP,
homeostasis, structural plasticity, metabolism) models one creature's brain;
evolution across generations is a separate "life" layer. Constraint: **brains,
not robots** — competence must emerge from network dynamics/plasticity, never
hardcoded sensor-motor mappings. Reflex-intact and oracle controllers are
ceiling references only, never design options.

Repo: https://github.com/charlesdvaught-hash/DigitalOrganoid
Full findings log: `RESULTS.md` (Tasks 1-13)

## Infra
GPU evolution runs on Dave's own PC ("mothership", RTX 5070) via the
Desktop Commander bridge. Working dir, a real git checkout of `main`
tracking `origin`:
`C:\Users\charl\DigitalOrganoid-gpu\gpu_evolve.py`
Python invocation (bare `python` fails — Windows Store alias stub):
`& "C:\Users\charl\AppData\Local\Programs\Python\Python312\python.exe" gpu_evolve.py ...`
Launch in background via `start_process`, redirect with PowerShell `*>`,
poll with `read_file`/tail — long-running (~10-18 min for pop48/gens60,
train-seeds 8). Git installed via `winget install --id Git.Git` on the PC,
at `C:\Program Files\Git\cmd` (not always on PATH in a fresh PowerShell —
add it: `$env:Path += ";C:\Program Files\Git\cmd"`). The cloud sandbox
session is NOT authorized to push to this repo (git proxy 403) — commits
made there stay local; push from the PC checkout instead.

## Standard eval config
`--pop 48 --gens 60 --train-seeds 8 --heldout-seeds 8 --heldout-t 6000`,
no-reflex curriculum (reflex_scale fades 1.0→0.0 over the first
`--curriculum-gens` generations, default 40; held-out eval is always
reflex_scale=0.0 regardless of training args). `--train-seeds 8` (raised
from 4 during Task 13 — see below) is the number of food layouts sampled
per generation; too few lets evolution reward layout-specific luck instead
of generalizing. Primary metric: **steering correlation** (Pearson r,
sensor L-R asymmetry vs. motor L-R asymmetry), not food count alone — food
count rewards ANY path to food including non-directional "wander and bump
into it" strategies; steering r isolates genuine gradient-following. Always
check both. Replication protocol: any promising result on scaffold-seed 42
gets rerun on `--scaffold-seed 7` before being trusted.

## Current best mechanism: FEP / predictability learning (Task 9)
`--learning fep` (plain unmodulated Hebbian STDP, no dopamine gating) +
`--fep-punish` (wall proximity or too-long-without-food triggers a burst of
pure noise replacing the smell/eye signal — DishBrain/bl1-inspired: no new
weight rule, credit assignment lives entirely on the environment side).
Defaults: `--fep-punish-t 150 --fep-wall-thresh 0.7 --fep-timeout-steps 800`.
- seed 42: 7.75±2.07 food, steering r=**0.080** (6/8 positive)
- seed 7 (replication): 10.62±2.08 food, steering r=**0.041** (5/8 positive)
- Beats every mechanism tried since (Tasks 10-13), on steering correlation.
  Nothing has come close to matching it yet, let alone beating it.

## Quick rundown of everything tried so far (all vs. this baseline)

| Task | Mechanism | Food | Steering r | Verdict |
|---|---|---|---|---|
| 9 | FEP + punishment (**baseline**) | 7.75±2.07 | **0.080** | — |
| 10 | FEP-punish + `surprise` rule combined | — | -0.031 / 0.002 | worse — don't combine |
| 11 | relaxed punishment (softer, 3 variants) | 5.4-10.8 | 0.008-0.016 | worse, punishment intensity matters |
| 12 | harsher punishment (all 3 params up) | 5.25±1.55 | 0.006 | worse — intensity bracketed both sides |
| 13a | `--homeo sub` (subtractive normalization) | 6.12±1.67 | -0.008 | worse |
| 13b | `--learning eh` (Exploratory Hebbian) | 5.12±1.55 | 0.021 | worse, but more genuine asym |
| 13c-i | `--fep-punish-gate-steps 1000` (timing gate) | 6.12±1.46 | 0.006 | worse |
| 13c-ii | critical-period gating (`--cp-*`) | 5.12±1.27 | -0.022 | worse on both metrics |
| 13c-iii | standalone `--learning btsp` (event-gated elig.) | 6.12±1.01 | 0.014 | worse, but 7/8 positive (most consistent sign) |

**Diagnosis behind Task 13** (from `CANDIDATE_MECHANISMS.md`): the plasticity
rule is left/right symmetric — every `sL` neuron gets identical injection,
both motor pools are fed by the same K-NN graph, plain Hebbian potentiates
`sL->mL`/`sL->mR` equally — so no amount of better credit assignment can by
itself build a differential steering pathway; the only asymmetry available
is chance topology. A permanent `asym` weight-space diagnostic
(`mean(w[sL->mL]) - mean(w[sL->mR])`, printed every generation and for the
final champion) was added in Task 13 to measure this directly. None of the
symmetry-breaking or timescale/gating fixes tried (items 1-3 of the
shortlist) recovered steering above baseline, though EH (13b) and standalone
btsp (13c-iii) showed weak positive movement worth remembering if candidate
#4 also comes up short.

Full per-task detail, per-seed numbers, and run-artifact filenames are in
`RESULTS.md`.

## Immediate next step: candidate #4 — evolve-the-plasticity-rule (not started)
Najarro & Risi-style: instead of evolving ~6000 per-synapse weights directly
(what every mechanism above still does), evolve a much smaller genome — per
`(pre_pool, post_pool)` pair Hebbian rate coefficients (roughly ~80-200
values depending on how many distinct pool-pairs the topology actually has,
vs. ~6000 synapses) — that each individual's own lifetime Hebbian/STDP
dynamics then expresses into synaptic weights from a fixed/shared starting
point. This directly targets the fitness-gaming/overfitting axis Dave
flagged in Task 13 (evolution memorizing layout-specific synaptic patterns)
by shrinking what evolution can actually overfit to.

Sketch of the implementation (scoped but not written):
1. `build_pair_map(scaffold)`: for every plastic synapse, tag it with a
   `(pre_pool, post_pool)` pair (pools = the existing `POOLS` dict plus a
   catch-all `'hid'` for interneurons); assign each distinct pair observed
   in the topology an index. This becomes a new `pair_idx` tensor on
   `BatchSim`, plus `n_pairs`.
2. New `--learning evolverule` mode: genome becomes
   `2 * n_pairs` values (`A_PLUS`/`A_MINUS` per pair, in place of per-synapse
   weights). Bounds/init need their own CLI knob (e.g. `--rule-rate-max`).
3. Synaptic weights (`wp`) no longer come from the genome — they start from
   a fixed, shared initial value (the scaffold's own baked-in init weight)
   and evolve during the lifetime via plain Hebbian STDP, same math as the
   `fep` branch, but with `A_PLUS`/`A_MINUS` gathered per-synapse from the
   evolved per-pair genome instead of the two global constants.
4. `run_lifetime()`/`main()` need a branch: for `evolverule`, the "genome
   batch" passed around is the rule-coefficient array, not the weight array
   — construct the actual `wp` tensor from `sim`'s fixed init weight
   separately, and pass the rule coefficients through to `sim.run()` as new
   kwargs (`rule_a_plus`, `rule_a_minus`).
5. The `asym` diagnostic doesn't apply directly to the genome anymore (it's
   not a weight array) — needs to run on the *expressed* weights after a
   lifetime instead (e.g. captured from the held-out champion's final `wp`).

None of this is committed to `gpu_evolve.py` yet — start here.

## Lower-priority backlog, not yet touched
- **FEP + eye pool** (`--use-eye`) — untested. Task 8's null result on
  vision used reward-gated `stdp` mode; may not generalize to FEP's
  plain-Hebbian rule.
- **Third-topology replication** for the Task 9 baseline itself (currently
  2/2 positive across seed 42 and seed 7) — more confidence before
  investing further in this mechanism.
- **Scale-up** (bigger pop/more gens) on the Task 9 FEP baseline — unknown
  if it plateaus or keeps climbing past pop48/gens60.
- Other GitHub branches (`creature-spiking-sweep-*`, `main.-Evo`,
  `evolution-experiments-prototype-*`, `feature-realistic-spiking-organoid-*`)
  hold older, unmerged experiments (`brain.js`/`evolution_arena.html`,
  `evolution_experiments.py`'s inheritance operators) — see RESULTS.md's
  "Generalization track" sections. `brain.js` was pulled onto a working
  branch and given a `synGain` fix but never passed the steering-correlation
  gate; not validated, don't build on it yet.
