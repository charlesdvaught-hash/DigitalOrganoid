# DigitalOrganoid — session handoff (Aug 14 2026)

## Project
"Digital life simulator": Izhikevich SNN brain (Dale's principle, STDP,
homeostasis, structural plasticity, metabolism) models one creature's brain;
evolution across generations is a separate "life" layer. Constraint: **brains,
not robots** — competence must emerge from network dynamics/plasticity, never
hardcoded sensor-motor mappings. Reflex-intact and oracle controllers are
ceiling references only, never design options.

Repo: https://github.com/charlesdvaught-hash/DigitalOrganoid
Full findings log: `RESULTS.md` (Tasks 1-11)

## Infra
GPU evolution runs on Dave's own PC ("mothership", RTX 5070) via the
Desktop Commander bridge. Working dir:
`C:\Users\charl\DigitalOrganoid-gpu\gpu_evolve.py`
Python invocation (bare `python` fails — Windows Store alias stub):
`& "C:\Users\charl\AppData\Local\Programs\Python\Python312\python.exe" gpu_evolve.py ...`
Launch in background via `start_process`, redirect with PowerShell `*>`,
poll with `read_file`/tail — long-running (~10-18 min for pop48/gens60).
Sandbox copy at `/tmp/DigitalOrganoid/gpu_evolve.py`; keep both in sync
(edit sandbox → SendUserFile → device_commit_files to the PC path).

## Standard eval config
`--pop 48 --gens 60 --fixed-reflex-scale 0.0 --heldout-seeds 8 --heldout-t 6000`
Primary metric: **steering correlation** (Pearson r, sensor L-R asymmetry vs
motor L-R asymmetry), not food count alone — food count rewards ANY path to
food including non-directional "wander and bump into it" strategies; steering
r isolates genuine gradient-following. Always check both. Replication
protocol: any promising result on scaffold-seed 42 gets rerun on
`--scaffold-seed 7` before being trusted (surprise-learning's strong result
failed this check; FEP passed it).

## Current best mechanism: FEP / predictability learning (Task 9)
`--learning fep` (plain unmodulated Hebbian STDP, no dopamine gating) +
`--fep-punish` (wall proximity or too-long-without-food triggers a burst of
pure noise replacing the smell/eye signal — DishBrain/bl1-inspired: no new
weight rule, credit assignment lives entirely on the environment side).
- seed 42: 7.75±2.07 food, steering r=**0.080** (6/8 positive)
- seed 7 (replication): 10.62±2.08 food, steering r=**0.041** (5/8 positive)
- Beats every prior no-reflex mechanism's food mean; only mechanism whose
  steering signal held positive across two topologies.

## Tried and ruled out
- **Task 10**: combining FEP-punish with the `surprise` (eligibility×TD-error)
  learning rule instead of plain Hebbian — makes steering WORSE (r=-0.031 and
  0.002 across two variants), not better. Don't combine these two.
- **Task 11**: relaxing punishment params (shorter burst t=60, longer
  timeout=1500, looser wall=0.85) all traded steering for raw food —
  baseline (t=150/wall=0.7/timeout=800) beats every relaxed variant's
  steering by 5-10x. Punishment *intensity* is doing the work.

## Immediate next step (not yet run)
**Harsher punishment sweep** — the untested opposite direction from Task 11:
shorter `--fep-timeout-steps` (e.g. 400), lower `--fep-wall-thresh` (e.g.
0.5), longer `--fep-punish-t` (e.g. 300). Evaluate on steering r as primary
metric. If a harsher setting beats 0.080, replicate it on seed 7 before
trusting it (same protocol as Task 9).

## Other open threads, lower priority
- FEP + eye pool (`--use-eye`) untested — Task 8's null result on vision used
  reward-gated `stdp` mode, may not generalize to FEP's plain-Hebbian rule.
- Third-topology replication for FEP (currently 2/2 positive).
- Scale-up (bigger pop/more gens) on FEP baseline, untested — unknown if it
  plateaus or keeps climbing past pop48/gens60.
