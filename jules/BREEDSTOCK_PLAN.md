# Breedstock plan — evolve the creature's brain

Next-phase spec. Start this in a fresh thread; the embodied sweep
(`EMBODIED_SWEEP.md`) is complete and is the foundation this builds on.

## Decision
Evolve the **synaptic weights** (the trained brain), not just the
hyperparameters. Offspring inherit a brain, so selection compounds across
generations — real breedstock, not just good body plans.

## What the sweep already told us (use as fixed scaffold)
Robust main effects over 324 runs (rankings of individual cells were too noisy —
food_sd ≈ food_mean at 3 seeds — so trust these averages, not any single winner):
- **N = 600** is the sweet spot: best foraging efficiency (0.425 food/1k steps)
  AND longest survival (~4,500 steps). 200 too small, 1200 starves sooner.
- **K = 6** best; higher K hurts survival.
- **Dale = True** modestly better on both axes.
- **fi** (inhibitory %) ~irrelevant across 0.1–0.3; use 0.2.
- **STDP** had ~no behavioral effect (0.343 vs 0.342) — reward is too rare at
  this scale to shape motors. Implication: don't rely on lifetime learning for
  fitness; evolution must do the optimizing.

**Fixed genome scaffold:** N=600, K=6, fi=0.2, dale=True. Evolve the weights on
top of that. (Optionally keep stdp on as cheap within-life plasticity.)

## Build outline
1. **Genome = weight vector.** Reuse `creature_embodied.build_creature_network`
   for topology (pre/post/is_reflex fixed), but make the plastic-synapse weights
   the genome. Reflex weights stay innate/fixed.
2. **Fitness (stable ranking):** evaluate each genome over **≥15 seeds** (fresh
   arenas), score = mean of a composite, e.g.
   `fitness = food_eaten + 0.0003*survival_steps` (tune weights). Report mean±sd;
   low variance is the whole point.
3. **Selection + breeding:** keep top ~20%; produce offspring by
   (a) crossover — per-synapse pick from two parents, and
   (b) mutation — Gaussian noise on plastic weights, respecting Dale sign clamps
   and W_MAX. Elitism: carry the champion unchanged.
4. **Generational loop** with checkpointing (same resumable pattern as the
   sweep — the workspace kills long background jobs, so use a serial batch driver
   like `run_batch.py`, ~30–40s per call).
5. **Champion store:** save top genomes to JSON (weights + provenance) as the
   default/breedstock pool. These become selectable defaults in `index.html`.
6. **(Later) competition arena:** multiple champions, shared food, head-to-head
   ranking as an alternative fitness — reuse the same body/sensor code.

## Open questions to settle at the start of the new thread
- Fitness weighting: pure forager (food) vs. survivor (energy/longevity) vs. blend.
- Population size / generations vs. runtime budget (each genome = 15+ sims).
- How champions plug back into `index.html` as default brains.
