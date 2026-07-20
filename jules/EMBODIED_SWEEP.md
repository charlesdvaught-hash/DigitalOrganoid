# Embodied creature sweep

The original `organoid_simulator/creature_model.py` sweep is a **noise-driven
abstraction** — its own docstring says *"No arena/body/3D migration here —
driven by random input."* It reports firing/avalanche/weight stats on a spiking
network fed i.i.d. noise, not on the creature.

This adds a sweep that actually **matches the creature** — a faithful offline
port of the embodied closed loop in `index.html`.

## Files

- `organoid_simulator/creature_embodied.py` — the port. `build_creature_network`
  (distance-based K-nearest connectivity on a unit-ball embedding + innate reflex
  links) and `run_creature_simulation` (the full loop below), vectorized in NumPy.
- `organoid_simulator/creature_embodied_sweep.py` — parameter sweep, same
  structure as `creature_sweep.py` (parallel small-N, budget-gated large-N,
  checkpointed/resumable). Writes `creature_embodied_sweep_checkpoints.json`.
- `tests/test_creature_embodied.py` — 9 unit tests (topology, Dale's law,
  reflexes fixed, STDP bounds, directional sensors, behavior metrics).
- `run_batch.py` — serial, resumable batch driver (safe to call repeatedly).

## The closed loop (faithful to index.html)

`sensors()` → 6 sensory pools (sL/sR food direction, sW wall, sD food-distance,
sH hunger, sT fatigue), injected as current at gain `Sg·14` → `stepBrain()`
(Izhikevich RS/IB/FS/LTS, ±8% jitter, Dale's law, reward-modulated STDP,
homeostatic scaling every 500 steps) → motor pools mL/mR → differential-drive
body in a 2D arena (speed from total motor firing, turn from L−R difference +
proportional wall avoidance, heading inertia) → metabolism
(`energy -= 0.030 + 2.0·speed + 400·speed²`, the U-shaped cost of transport) →
eating (+28 energy, eases fatigue, dopamine burst) → dopamine modulates STDP.

`stdp_on` gates plasticity + reward modulation; `N,K,fi,dale,seed` as before.

### Metrics recorded
Neural: `mean_pop_rate, pop_rate_cv, per_neuron_rate_std, num_avalanches,
mean/max_avalanche_size, init/final_mean_abs_weight, weight_drift,
mean_final_energy, mean_final_fatigue`.
Behavior (new): `food_eaten, distance_traveled, mean_speed, final_body_energy,
survival_steps, alive_end, food_per_1k_steps, food_per_distance`.

## Deliberate simplifications
Topology is **fixed** so the loop vectorizes and sweeps at speed. Structural
plasticity from index.html — synaptic pruning, reward-triggered synapse growth,
and morphogen/migration self-organization — is excluded. Synaptic (weight)
plasticity is fully retained. Default `T = 6000` ticks.

## Result
324 configs (N∈{200,600,1200} × K∈{6,10,16} × fi∈{0.1,0.2,0.3} × dale × stdp ×
3 seeds), all `success`. Sanity: STDP-on gives mean weight_drift ≈ 0.08 vs
**exactly 0** with STDP off; creatures starve at different times
(survival 3,112–6,000 steps); foraging varies 0–8 food, peaking at N=600 — none
of which the noise-driven sweep could produce.

Reproduce: `python3 -m organoid_simulator.creature_embodied_sweep`
(or `python3 run_batch.py 40` repeatedly).
