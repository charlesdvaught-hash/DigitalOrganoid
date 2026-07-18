# Structural / Init Variants to Try (Phase 0.5)

Goal: run the *same* engine and metrics, swap one "starting calculation" at a time, and see
which produces more functional potential — higher memory capacity (MC), genuine criticality
(σ→1, statistically significant power-law), and — newly important — **less global synchrony**
(more differentiated local activity instead of the whole blob pulsing with the shared input).

For every variant, log: MC, avalanche exponent + power-law significance (LLR, p), branching
ratio σ, the structural descriptors (clustering coeff, mean path length, degree variance,
modularity, weight-distribution skew, mean connection distance), AND a **synchrony metric**
(fraction of total variance explained by the population common-mode, or mean pairwise
correlation) so we can tell which variants break the unison pulsing.

## Variants (priority order)

1. **Non-negative activation — "functional Dale's law" (HIGHEST PRIORITY).**
   Replace tanh with a non-negative activation (ReLU, or a shifted/rectified tanh mapping to
   [0,1]) so a neuron's output is always ≥0 and the fixed weight sign alone sets its effect.
   *Why:* measured result — with tanh, inhibitory neurons are in a negative state ~50% of the
   time, so they push *excitatory* ~half the time and net out to ≈0. Inhibition is currently
   cosmetic. Fixing it should restore real E/I balance, and is the most likely single change to
   break global synchrony and lift MC / enable criticality.

2. **Small-world (Watts–Strogatz):** ring lattice + random rewiring of a fraction p of edges.
   Brain-like: high clustering + short path length. Sweep p (e.g. 0.01, 0.05, 0.1).

3. **Scale-free / hub (Barabási–Albert):** power-law degree, a few high-degree hubs.
   Most likely topology to yield genuine power-law avalanches / criticality.

4. **Log-normal weights:** keep wiring, draw connection *strengths* from a log-normal (a few
   strong, many weak) instead of half-normal. Cheapest change; strong cortical realism.

5. **Modular / clustered:** dense within communities, sparse between. Promotes functional
   modules and directly opposes global synchrony.

6. **3D spatial topology:** neurons at random 3D positions in a ball, wired by physical
   distance-decay (already prototyped in `organoid_viewer_3d.html`). More organoid-faithful
   than the 1D ring; keep as the spatial baseline.

## Then: design a 7th from the winners
Once 1–6 are run, correlate the structural descriptors + activation choice against MC /
criticality / synchrony, identify which *features* drive the gains, and hand-design a hybrid
that combines the winning features (e.g. non-negative activation + small-world clustering +
log-normal weights + hubs). If that hybrid validates, it justifies automating the search
(the Phase 2 breeding loop) instead of hand-picking one combination at a time.

## New metric to add to the pipeline
- **Synchrony / common-mode fraction:** at each timestep the population mean is the "common
  mode"; report what fraction of total state variance it accounts for. High = input-echoing
  (bad); low = distributed local computation (good). This is the quantitative version of the
  "blinking in unison" we saw in the viewer.
