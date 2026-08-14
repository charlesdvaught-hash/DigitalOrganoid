# DigitalOrganoid — session handoff (Aug 14 2026)

## Project
"Digital life simulator": Izhikevich SNN brain (Dale's principle, STDP,
homeostasis, structural plasticity, metabolism) models one creature's brain;
evolution across generations is a separate "life" layer. Constraint: **brains,
not robots** — competence must emerge from network dynamics/plasticity, never
hardcoded sensor-motor mappings. Reflex-intact and oracle controllers are
ceiling references only, never design options.

Repo: https://github.com/charlesdvaught-hash/DigitalOrganoid
Full findings log: `RESULTS.md` (Tasks 1-12)

## Infra
GPU evolution runs on Dave's own PC ("mothership", RTX 5070) via the
Desktop Commander bridge. Working dir, now a real git checkout of `main`
tracking `origin`:
`C:\Users\charl\DigitalOrganoid-gpu\gpu_evolve.py`
Python invocation (bare `python` fails — Windows Store alias stub):
`& "C:\Users\charl\AppData\Local\Programs\Python\Python312\python.exe" gpu_evolve.py ...`
Launch in background via `start_process`, redirect with PowerShell `*>`,
poll with `read_file`/tail — long-running (~10-18 min for pop48/gens60).
Git itself was not installed on the PC or in the cloud sandbox as of Aug 14
2026; installed via `winget install --id Git.Git` on the PC. The cloud
sandbox session is NOT authorized to push to this repo (git proxy 403) —
commits made there stay local; push from the PC checkout instead, using
whatever GitHub credentials/credential-manager prompt comes up there.

## Standard eval config
`--pop 48 --gens 60 --heldout-seeds 8 --heldout-t 6000`, no-reflex curriculum
(reflex_scale fades 1.0→0.0 over the first `--curriculum-gens` generations,
default 40; held-out eval is always reflex_scale=0.0 regardless of training
args). Primary metric: **steering correlation** (Pearson r, sensor L-R
asymmetry vs. motor L-R asymmetry), not food count alone — food count rewards
ANY path to food including non-directional "wander and bump into it"
strategies; steering r isolates genuine gradient-following. Always check
both. Replication protocol: any promising result on scaffold-seed 42 gets
rerun on `--scaffold-seed 7` before being trusted (surprise-learning's strong
result failed this check; FEP passed it).

## Current best mechanism: FEP / predictability learning (Task 9)
`--learning fep` (plain unmodulated Hebbian STDP, no dopamine gating) +
`--fep-punish` (wall proximity or too-long-without-food triggers a burst of
pure noise replacing the smell/eye signal — DishBrain/bl1-inspired: no new
weight rule, credit assignment lives entirely on the environment side).
Defaults: `--fep-punish-t 150 --fep-wall-thresh 0.7 --fep-timeout-steps 800`.
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
- **Task 12** (Aug 14 2026): pushed the same three params HARSHER instead —
  `--fep-timeout-steps 400 --fep-wall-thresh 0.5 --fep-punish-t 300` — also
  worse (food 5.25±1.55, r=0.006, seed 42), in fact the worst steering result
  on record, tied with Task 11's looser-wall variant. Punishment intensity is
  bracketed on both sides now, not monotonic: baseline (150/0.7/800) sits at
  or near a local optimum, not one end of a "more is better" ramp. Not
  replicated on seed 7 (didn't beat baseline, so didn't meet the replication
  bar).

## Immediate next step (not yet run)
With punishment intensity bracketed on both sides (Task 11: too soft loses,
Task 12: too harsh loses), reasonable next moves, roughly in priority order:
1. **Fine sweep around baseline** — small perturbations to one param at a
   time near 150/0.7/800 (not the large jumps Tasks 11-12 used) to map the
   shape of the local optimum, rather than assuming it's a single point.
2. **FEP + eye pool** (`--use-eye`) — untested. Task 8's null result on
   vision used reward-gated `stdp` mode; may not generalize to FEP's
   plain-Hebbian rule.
3. **Third-topology replication** for the Task 9 baseline itself (currently
   2/2 positive across seed 42 and seed 7) — more confidence before
   investing further in this mechanism.
4. **Scale-up** (bigger pop/more gens) on the Task 9 FEP baseline — unknown
   if it plateaus or keeps climbing past pop48/gens60.

## Other open threads, lower priority
- Other GitHub branches (`creature-spiking-sweep-*`, `main.-Evo`,
  `evolution-experiments-prototype-*`, `feature-realistic-spiking-organoid-*`)
  hold older, unmerged experiments (`brain.js`/`evolution_arena.html`,
  `evolution_experiments.py`'s inheritance operators) — see RESULTS.md's
  "Generalization track" sections. `brain.js` was pulled onto a working
  branch and given a `synGain` fix but never passed the steering-correlation
  gate; not validated, don't build on it yet.
