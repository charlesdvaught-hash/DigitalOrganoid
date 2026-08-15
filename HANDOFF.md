# DigitalOrganoid — session handoff (Aug 14 2026)

## Where things stand right now
Shortlist items 0-4 of `CANDIDATE_MECHANISMS.md` are now all closed out,
plus one Dave-proposed candidate (evo-devo guidance-code wiring). **None
beat the Task 9 FEP baseline's steering correlation (r=0.080/0.041, seed
42/seed 7).** Task 9 remains the best-validated mechanism on record.

**Candidate #4 (evolve-the-plasticity-rule)** roughly doubles food count on
both topologies (huge effect, biggest of the project) but does NOT reliably
build steering correlation — seed 42 showed strong negative r (-0.267) that
did not replicate on seed 7 (r=-0.035, noise). Adding a direct r-bonus to
training fitness fixed r on training seeds but did not transfer to held-out
seeds (overfitting).

**Candidate A (evo-devo guidance-code wiring, Dave's idea)** — one data
point so far, below baseline on both food and r; inconclusive, not
disproven (low-dimensional genome may just need more generations).

**Big open question this session tried and failed to answer:** is
candidate #4's food-count gain genuine hunting behavior or a "lucky
wanderer" effect? Built four increasingly careful behavioral metrics
(`approach_frac` → 200-step net-approach → `long_range_correlation` →
`hunt_score` → `hunt_score_v2`); the first was tautological (caught by
Dave), and the last two — the ones meant to be rigorous — both failed a
null control: **random, untrained, no-learning weights score the same as
every evolved champion**, including the Task 9 baseline. Conclusion:
steering correlation remains the ONLY metric in this project that reliably
discriminates trained from untrained behavior. Do not trust food count, or
any of the distance/hunt-based metrics built this session, as evidence of
real foraging competence without also checking steering r. Full
investigation, per-metric detail: RESULTS.md Task 16.

**Task 17 — new standard diagnostic, and it explains Task 16's mystery:**
built `bench_diode()` per Dave's framing ("model it like a microcontroller
— test the diode before plugging it into the robot"): clamp smell L/R to a
fixed value, freeze the body, learning off, read steady-state mL-mR. No
embodiment confound possible. Decomposes response into BIAS (turn tendency
at symmetric input — a built-in spin, unrelated to sensing) and SLOPE (real
gradient-tracking). **Candidate #4's expressed weights carry a large,
highly reliable BIAS (-0.15 on seed7, +0.04 on the rfit/seed42 variant,
replicated 3/3 independent lifetimes each) and a noise-level SLOPE** — this
explains its whole profile: a constant turn sweeps more arena (→ the food
gain) while contributing ~0 to a metric that needs live tracking (→ the
noisy/near-zero steering-r). Task 9 baseline and candidate A show neither a
reliable bias nor slope at this bench resolution. **Use `bench_diode()` as
the default first check on any new candidate going forward**, alongside
food/steering-r. Full detail: RESULTS.md Task 17.

RESULTS.md and this file are up to date and committed locally on the PC —
**push to GitHub still needed**, the cloud session can't push (403, repo
not in its authorized set); push from the PC checkout when convenient.

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

**As of Task 17, also run `bench_diode()` on any new candidate's champion**
(RAW genome and EXPRESSED post-lifetime weights, ideally 2-3 independent
lifetime replicates to check the result isn't a fluke of one random life).
It separates a fixed motor BIAS (spurious, inflates food count without real
steering) from genuine SLOPE (asymmetry-tracking) in a way food count and
steering-r alone can't — see RESULTS.md Task 17 for why this matters.

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
| 14 | `--learning evolverule` (candidate #4), seed42 | 12.62±2.20 | -0.267 | food ~2x, r doesn't replicate |
| 14 | `--learning evolverule`, seed7 (replication) | 18.12±2.48 | -0.035 | negative r was topology-specific noise |
| 14fu | evolverule + r-aware fitness bonus | 17.00±3.78 | -0.038 | r-bonus overfits to training seeds |
| 15 | `--wiring guide` (candidate A, Dave's idea) | 2.25±0.70 | 0.002 | 1 data point, inconclusive |
| 16 | behavioral metrics for #4's food gain | — | — | all 4 tried are tautological or fail null control |
| 17 | `bench_diode()` — isolated open-loop probe | — | — | candidate #4 = large constant motor bias, not steering |

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

## Immediate next step: not decided yet — options on the table
No committed direction. Candidates, roughly in order of how much groundwork
is already done:
1. **Push candidate A (wiring-guide) further** — only 1 data point (Task
   15), below baseline, but genuinely novel and low-dimensional (30 genes);
   more generations or a gain sweep before ruling it out.
2. **Third-topology replication of Task 9 baseline** — currently 2/2
   positive (seed 42, seed 7); a seed-13 or similar run would raise
   confidence it's not itself a topology-specific fluke.
3. **Scale-up Task 9 baseline** — bigger pop/more generations, unknown if
   it plateaus or keeps climbing past pop48/gens60.
4. **FEP + eye pool** (`--use-eye`) — untested; Task 8's null result used
   `stdp` mode, may not generalize to FEP's plain-Hebbian rule.
5. **Try to fix candidate #4's bias-vs-slope imbalance directly** — now
   that Task 17 shows *why* it doesn't steer (huge bias, no slope), a
   natural next test is whatever might suppress the bias term specifically
   (e.g. a symmetry constraint on the evolved rule genome, or bench-testing
   mid-evolution to select against high-bias/low-slope champions).

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
