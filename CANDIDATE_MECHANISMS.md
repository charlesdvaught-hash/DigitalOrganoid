# Candidate learning mechanisms for the no-reflex steering gap

Research memo, Aug 14 2026. No code changed. Baseline to beat: Task 9 FEP
(`--learning fep --fep-punish`), steering r = 0.080 (seed 42) / 0.041 (seed 7).
Reference ceilings (not design targets): reflex-intact r ≈ 0.175, oracle food ≈ 26-27.

---

## Part 1 — Diagnosis: what the code says the bottleneck is

Before the literature, three structural facts read directly out of `gpu_evolve.py`.
They matter because they change which candidates are plausible, and because every
mechanism tried so far (Tasks 4-12) leaves all three untouched.

### 1.1 The plasticity rule is left/right symmetric, so it cannot build a steering map

`sL` (neurons 0-19) all receive the *identical* scalar injection `L * sensor_factor`;
`sR` (20-39) likewise. `mL` (120-154) and `mR` (155-189) are both fed by the same
distance-based K-NN recurrent graph. Steering requires
`w(sL→mL) > w(sL→mR)` — a *differential* between two pathways.

Plain Hebbian STDP (`learning='fep'`, lines 412-423) potentiates a synapse whenever
pre and post co-fire. When `sL` is active and both motor pools are firing, **both**
`sL→…→mL` and `sL→…→mR` paths are potentiated by the same expected amount. There is
no term anywhere in the rule that can push one up while pushing the other down for
the same input. The only asymmetry available is whatever chance the random 3D
K-NN scaffold happens to supply — which is exactly why results are topology-specific
(surprise learning: r=0.063 on seed 42, r=0.008 on seed 7) and why every mechanism
plateaus at 20-40% of ceiling.

**This is the classical result of [Miller & MacKay 1994](https://direct.mit.edu/neco/article/6/1/100/5780/The-Role-of-Constraints-in-Hebbian-Learning):
Hebbian learning under *multiplicative* constraints preserves the initial symmetry;
only *subtractive* constraints produce winner-take-all competition and symmetry
breaking (this is how ocular dominance segregation works).** The homeostasis at
lines 447-456 is multiplicative (`wp = wp * sf`) — the form that provably does *not*
break symmetry.

### 1.2 The compensatory mechanism is ~20x too slow and the wrong form

Homeostasis fires every 500 steps and moves weights by ±5%. `A_PLUS = 0.008` with
`W_MAX = 2.5` means a synapse can traverse its full range in ~300 coincidence events —
well inside one homeostatic interval at realistic firing rates.
[Zenke & Gerstner 2017](https://royalsocietypublishing.org/rstb/article/372/1715/20160259/23102/Hebbian-plasticity-requires-compensatory-processes)
show analytically that homeostatic compensation slower than the Hebbian timescale
cannot stabilise learning at all — runaway potentiation wins first, and what you
observe is a saturated, undifferentiated weight matrix. Their prescription: fast
compensatory processes (heterosynaptic depression, BCM sliding threshold) on the
*same* timescale as the Hebbian term.

Both 1.1 and 1.2 point at the same fix and neither has been tried.

### 1.3 The credit-assignment timescale doesn't match the behavioural loop

`TRACE_DECAY = 0.96` → ~25-step STDP horizon. The `surprise` rule extends this to
~200 steps. But the actual behavioural loop here — turn, travel, arrive, eat — is
bounded below by `speed ≈ 0.001-0.01` units/step across a unit arena, i.e. hundreds
to low thousands of steps. Both existing traces are shorter than the causal chain
they're supposed to credit.
[Bittner et al. 2017 (Science)](https://www.science.org/doi/10.1126/science.aan3846)
and the [Gerstner et al. 2018 review](https://arxiv.org/abs/1801.05219) document
eligibility traces on exactly these behavioural timescales (seconds, ~1-10 s), which
is the regime this simulation is in.

**Practical implication for every experiment below:** add a direct weight-space
diagnostic alongside steering r —
`asym = mean(w[sL→mL]) - mean(w[sL→mR])` (and the sR mirror), logged per generation.
Steering r is a noisy behaviour-level readout; `asym` measures the thing the mechanism
is *supposed* to be building, and would tell you within one run whether a candidate
differentiates the pathways at all. ~15 lines, near-zero runtime cost. Do this first
regardless of which mechanism comes next.

---

## Part 2 — Candidates

Cost is stated as (new/changed lines in `gpu_evolve.py`) + (standard eval runs at
pop48/gens60 ≈ 10-18 min each, seed 42 then seed 7).

---

### A. Rules that break L/R symmetry by construction

#### A1 — Exploratory Hebbian (EH) / reward-modulated Hebbian with double baseline subtraction

**Source:** [Legenstein, Chase, Schwartz & Maass 2010, *J. Neurosci.* 30(25):8400](https://www.jneurosci.org/content/30/25/8400);
[Hoerzer, Legenstein & Maass 2014, *Cerebral Cortex* 24(3):677](https://academic.oup.com/cercor/article/24/3/677/392266).

**Mechanism.** `Δw_ij = η · [R(t) − R̄(t)] · x_j(t) · [a_i(t) − ā_i(t)]`, where `R̄` and
`ā_i` are low-pass-filtered running means of reward and of postsynaptic activity
(the paper uses `z̄(t) = 0.8·z̄(t−1) + 0.2·z(t)`). The `[a_i − ā_i]` term is the
critical part: it is a *signed* quantity that isolates the neuron's own spontaneous
fluctuation away from its baseline, so the rule potentiates synapses onto neurons
that happened to fire *more* than usual when reward beat expectation, and depresses
synapses onto neurons that fired *less*. Hoerzer et al. show this alone turns a
chaotic recurrent network into a functioning controller from a scalar reward with no
teacher signal.

**Why it's the top pick here.** This is precisely the missing term from §1.1. For one
sensory input it can drive `mL` up and `mR` down *simultaneously*, because their
fluctuations from baseline have opposite signs on any given step. Plain STDP, R-STDP,
and the `surprise` rule all lack this: `surprise` uses `elig × (dopamine − value)`,
which subtracts a baseline from *reward* but never from *postsynaptic activity* — so
it is still symmetric across the two motor pools. **This is not "R-STDP again"** and
does not conflict with the Task 4 / Task 10 negative results.

**Codebase mapping.** New `--learning eh` branch alongside the existing three, ~30
lines. You already have `smoothed` (per-neuron low-pass activity) — add a slower
second filter `a_bar` (one `(B,N)` tensor, one line per step) and a `dopa_bar` scalar
per batch row. `x_j` = `smoothed[:, pre_p]`, `a_i − ā_i` = `smoothed[:, post_p] − a_bar[:, post_p]`.
Dale clamp applies unchanged at the end (same three lines as every other branch).
Izhikevich dynamics untouched. Reward: reuse the existing `dopamine` variable exactly
as `surprise` does.

**Cost.** ~30 LOC, 2 runs (seed 42 + seed 7 replication). ~40 min GPU.
**Risk.** The `η` scale needs one order-of-magnitude bracket; budget one throwaway
short run to pick it (validate reflex-intact first, as Task 6 did for `surprise`).

---

#### A2 — Subtractive normalisation + fast heterosynaptic depression (replace the current homeostasis)

**Source:** [Miller & MacKay 1994, *Neural Computation* 6(1):100](https://direct.mit.edu/neco/article/6/1/100/5780/The-Role-of-Constraints-in-Hebbian-Learning);
[Zenke & Gerstner 2017, *Phil. Trans. R. Soc. B* 372:20160259](https://royalsocietypublishing.org/rstb/article/372/1715/20160259/23102/Hebbian-plasticity-requires-compensatory-processes).

**Mechanism.** Instead of scaling all of a neuron's incoming weights by a common
factor, *subtract* a constant from all of them each update, so the total incoming
weight is conserved but individual synapses compete for a fixed budget: a synapse can
only grow if others shrink. Miller & MacKay prove this is what turns Hebbian
correlation-following into competitive selection. Zenke & Gerstner add that the
subtraction must run on the Hebbian timescale, not 500 steps behind it, and pair it
with weight-dependent heterosynaptic depression of unstimulated synapses during
high postsynaptic activity.

**Codebase mapping.** Surgical: replace lines 447-456. Compute per-post-neuron
`excess = index_add(|Δw| this step) / fan_in` and subtract it from every incoming
synapse of that neuron, every step (a second `index_add_` on `post_p`, same pattern
already used for `I_syn`). Add a `--homeo {mult,sub}` flag so the current behaviour
stays reproducible. Dale clamp already prevents sign flips. ~25 lines.

**Why it ranks high.** It is the cheapest possible change to the *current best*
mechanism (FEP), it targets the diagnosed failure directly, and it composes with
every other candidate here. Also the single most likely explanation for why every
mechanism tops out in the same 20-40% band regardless of what the learning rule does.

**Cost.** ~25 LOC, 2 runs. ~40 min GPU. Stackable with A1.
**Risk.** Subtractive rules can drive many weights to the clamp at zero; needs the
`asym` diagnostic from §1.3 to distinguish "healthy pruning" from "network went silent".

---

#### A3 — Inhibitory STDP for detailed excitation/inhibition balance

**Source:** [Vogels, Sprekeler, Zenke, Clopath & Gerstner 2011, *Science* 334:1569](https://www.science.org/doi/10.1126/science.1211095).

**Mechanism.** A symmetric spike-timing rule on *inhibitory* synapses drives each
excitatory neuron toward a target firing rate by matching local inhibition to local
excitation. The reported functional consequence is that balanced networks store
memories in a "silent" form that is unmasked when the matching input arrives —
i.e. inhibitory plasticity converts an undifferentiated blob into input-selective
assemblies.

**Codebase mapping.** Inhibitory synapses currently learn under the same STDP rule as
excitatory ones (the `exc_mask` only controls the clamp sign, lines 421-423). Add a
separate branch for `~exc_mask` synapses using the Vogels rule
(`Δw = η(pre·post_trace + post·pre_trace − ρ₀·pre)`), with `ρ₀` the target rate.
~20 lines; all tensors needed (`trace_pre`, `trace_post`, `firing_rate`) already exist.
`fi = 0.2` means 20% of the network is inhibitory, so there is real substrate here.

**Cost.** ~20 LOC, 2 runs. Best evaluated stacked on FEP, not alone.
**Note.** `EXPERIMENTS_TO_TRY.md` item 1 already flags "inhibition is currently
cosmetic" from the CPU-side measurement. This is the spiking-network version of that
fix and the two findings corroborate each other.

---

#### A4 — Intrinsic plasticity, and the full SORN triad

**Source:** [Triesch 2005, ICANN, "A gradient rule for the plasticity of a neuron's intrinsic excitability"](https://link.springer.com/chapter/10.1007/11550822_11);
[Lazar, Pipa & Triesch 2009, *Front. Comput. Neurosci.* 3:23 (SORN)](https://pubmed.ncbi.nlm.nih.gov/19893759/);
[Aswolinskiy & Pipa 2015, RM-SORN](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00036/full).

**Mechanism.** SORN combines three cheap local rules — STDP, synaptic normalisation,
and intrinsic plasticity (each neuron nudges its own threshold up after firing and
down when silent, toward a target rate) — and substantially outperforms static
reservoirs of the same size on sequence tasks, because the triad produces
well-separated internal states instead of one undifferentiated attractor. The RM-SORN
follow-up reports a directly relevant finding: **intrinsic plasticity is a better
source of exploration than injected noise**, and noise-driven learning failed
outright on their motion-generation task where IP-driven learning succeeded.

**Codebase mapping.** IP in an Izhikevich network = per-neuron adaptation of the
resting/reset parameter `c` or a per-neuron bias current, driven by the existing
`firing_rate` tensor: `bias += η_IP · (target_rate − firing_rate)`. One `(B,N)` tensor,
two lines in the step loop, add `bias` into `I`. ~15 lines. SN is A2. STDP exists.
So "SORN" here = A2 + A4, ~40 lines total.

**Relevance to the RM-SORN finding.** Task 7's bootcamp used annealed *motor noise*
for exploration and bought variance reduction but no ceiling gain. RM-SORN's result
predicts that swapping noise-based exploration for IP-based exploration is the
version that moves the ceiling. That is a concrete, testable prediction against an
already-collected null result.

**Cost.** ~15 LOC alone, ~40 LOC as the full triad. 2 runs.

---

### B. Credit assignment on the right timescale, with the right signal

#### B1 — Node / conductance perturbation (birdsong-style)

**Source:** [Fiete & Seung 2006, *Phys. Rev. Lett.* 97:048104](https://dx.doi.org/10.1103/PhysRevLett.97.048104);
[Fiete, Fee & Seung 2007, *J. Neurophysiol.* 98:2038](https://journals.physiology.org/doi/full/10.1152/jn.01311.2006).

**Mechanism.** Inject small independent random perturbations into each neuron's input
conductance, keep a per-synapse trace of `perturbation × presynaptic activity`, and
update weights by `trace × (reward − running mean reward)`. This is an unbiased
stochastic gradient estimate that needs no error signal, no backward pass, and no
knowledge of the network's structure — the songbird's LMAN does exactly this via
variability injection into RA.

**Distinct from what was ruled out.** R-STDP (Task 4) credits *pre/post coincidence*;
node perturbation credits *the injected exploratory fluctuation itself*. The
difference is the same one that makes A1 work: the perturbation term is signed and
independent per neuron, so it can differentiate `mL` from `mR`.

**Codebase mapping.** The scaffolding largely exists: `I_noise` (line 377) is already
a per-neuron independent perturbation, and `motor_noise_start/end` already implements
annealed exploration. Add a `(B, n_plastic)` perturbation-eligibility tensor
(same memory footprint as `elig` in the `surprise` branch — already proven to fit)
accumulating `I_noise[:, post_p] × smoothed[:, pre_p]`, and update with
`(dopamine − dopa_bar)`. ~35 lines.
**Caveat:** [Hiratani et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/cf38eb1549024cce4b3d2c1bb87a6c27-Paper-Conference.pdf)
show node perturbation's variance scales badly with network width — at N=600 with
sparse reward this may be the binding constraint. Pair with a denser reward proxy or
a longer trace.

**Cost.** ~35 LOC, 2 runs. Moderate risk of being too high-variance at this scale.

---

#### B2 — Behavioural-timescale eligibility (BTSP)

**Source:** [Bittner, Milstein et al. 2017, *Science* 357:1033](https://www.science.org/doi/10.1126/science.aan3846);
[Gerstner, Lehmann, Liakoni, Corneil & Brea 2018, *Front. Neural Circuits* 12:53](https://arxiv.org/abs/1801.05219);
[Cone & Shouval 2024/25 simple BTSP model, *Nat. Commun.*](https://www.nature.com/articles/s41467-024-55563-6).

**Mechanism.** A dendritic plateau event opens a seconds-long plasticity window that
retroactively potentiates *all* synapses active anywhere in that window — producing a
complete place field from a single traversal, no repetition needed. Unlike an
exponential eligibility trace, the window is gated by a discrete event and is roughly
flat across its span, so credit does not decay away over the approach.

**Codebase mapping.** Cheapest version: on an eat event (`hits > 0`), apply
`Δw ∝ trace_slow` where `trace_slow` is a very slow (decay ≈ 0.999, ~1000-step)
Hebbian coincidence trace — i.e. the `surprise` branch's `elig` tensor with a 5x
longer horizon and event-gated rather than continuous application. ~20 lines, mostly
reusing existing `surprise` machinery. This is a one-parameter-family generalisation
of something already implemented, which makes it very cheap to test.

**Why it's separate from Task 10's negative result.** Task 10 combined `surprise`
with `--fep-punish` and they interfered (punishment noise corrupts a continuously-applied
eligibility trace). An *event-gated* trace is applied only at eat events, which by
construction never coincide with a punishment window (punishment triggers on wall
proximity or food timeout). The stated interference mechanism does not apply.

**Cost.** ~20 LOC, 2 runs. Lowest-effort item on this list.

---

### C. Reward-free self-organisation from brain-body-environment coupling

#### C1 — Differential Extrinsic Plasticity (DEP)

**Source:** [Der & Martius 2015, *PNAS* 112(45):E6224](https://www.pnas.org/doi/10.1073/pnas.1508400112)
([arXiv:1505.00835](https://arxiv.org/abs/1505.00835)).

**Mechanism.** `τ Ċᵢⱼ = ỹᵢ ẋⱼ − Cᵢⱼ`, where `ẋⱼ` is the *time derivative* of sensor j,
`ỹᵢ = ẏᵢ + δẏᵢ` is the motor derivative corrected by a crude inverse model, and the
weight matrix is renormalised each step (`C ← κC/(‖C‖+ρ)`). Because the rule
correlates *changes* rather than *levels*, and is normalised, it amplifies exactly
those sensorimotor loops the body actually closes and starves the rest. The authors'
own framing of why this produces behaviour without any goal is
**"spontaneous symmetry breaking due to the tight brain-body-environment coupling"** —
which is verbatim the problem in §1.1. Coordinated crawling, rolling and object
manipulation emerge on 18-DOF robots within *minutes* of interaction, with no reward
and no per-system tuning.

**Codebase mapping.** Not a drop-in — DEP is formulated for rate units with
proprioceptive sensors, and your substrate is spiking with exteroceptive smell. The
faithful translation: apply DEP-form updates only to plastic synapses whose post is
in `mL`/`mR`, using `d(smoothed)/dt` for both pre and post (you already keep
`smoothed`; one extra `(B,N)` tensor holds the previous step), plus per-post-neuron
L2 renormalisation (the same `index_add_` pattern as A2). Skip the inverse model in
v1 — set `δẏ = 0` and rely on the derivative + normalisation, which is the
"differential Hebbian" core. ~45 lines.

**Why it's worth the cost.** It is the only candidate here whose published claim is
specifically *emergent sensorimotor competence with zero reward signal on a short
timescale* — the exact statement of your problem — and it requires no reward
plumbing at all, so it is orthogonal to every reward-shaping dead end in Tasks 4-12.

**Cost.** ~45 LOC, 2-3 runs (one to bracket `τ` and `κ`). ~60 min GPU.
**Risk.** Highest translation risk on this list; rate→spiking derivative estimation
is noisy. Mitigate by validating reflex-intact first.

---

#### C2 — Differential Hebbian / ICO learning

**Source:** [Porr & Wörgötter, ISO/ICO learning](https://www.berndporr.me.uk/isolearn/isolearning.pdf);
[Kolodziejski, Porr & Wörgötter 2009, *Biol. Cybern.*, TD-rules vs. differential Hebbian](https://pmc.ncbi.nlm.nih.gov/articles/PMC2798052/).

**Mechanism.** `Δw ∝ x · ẏ` — correlate a presynaptic signal with the *derivative* of
a later-arriving one, so an early predictive input gradually takes over the response
originally driven by a late "reflex" input. Provably related to TD learning, and
demonstrated producing taxis and avoidance in closed-loop embodied agents.

**Constraint caveat, stated plainly.** Classical ICO uses a reflex as the late signal.
That is disallowed here. The legitimate substitute is an *interoceptive* late signal
that is not a sensor→motor map: the derivative of `food_close` (line 336 — already an
existing sensor channel, `sD`). Using `d(food_close)/dt` as a third factor is reward
shaping, not sensorimotor hardcoding — but it is a real design decision and should be
called out in RESULTS.md if used, because it makes the reward denser than the current
eat-only dopamine and that alone could explain a gain.

**Cost.** ~25 LOC, 2 runs. Lower priority than C1 because of the constraint ambiguity.

---

#### C3 — Homeokinesis / predictive-information maximisation

**Source:** [Martius, Der & Ay 2013, *PLOS ONE* 8(5):e63400](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0063400);
Der & Martius, *The Playful Machine* (Springer 2012).

**Mechanism.** Drive learning by maximising the predictive information of the agent's
own sensor stream — the mutual information between past and future sensor values.
This is maximised by behaviour that is neither frozen nor random, which for an
embodied agent means coordinated, exploratory movement. No reward, no task.

**Assessment.** Conceptually the closest thing in the literature to what
`--fep-punish` is gesturing at, and a principled generalisation of it: instead of
scrambling sensors on failure (an environment-side hack), the agent directly optimises
sensory predictability. But computing predictive information online in a spiking
network is expensive and the published implementations are rate-based controllers.
**Listed for completeness; not recommended as a next prototype** — C1 is the same
research programme with a far cheaper local rule.

---

### D. Developmental scaffolding

#### D1 — Retinal-wave-style pre-behavioural structured spontaneous activity

**Source:** [Ge et al. 2021, *Science* 373:eabd0830, "Retinal waves prime visual motion detection by simulating future optic flow"](https://www.science.org/doi/10.1126/science.abd0830);
[Cang et al. 2005 / Chandrasekaran et al. 2005, retinotopic refinement requires waves in a critical period](https://www.cell.com/neuron/fulltext/S0896-6273(03)00790-6);
[Hebbian instruction of axonal connectivity by correlated spontaneous activity, *Science* 2023](https://www.science.org/doi/10.1126/science.adh7814).

**Mechanism.** Before eyes open, the retina generates spatiotemporally structured
spontaneous waves whose correlation structure mirrors the statistics of future
experience. Ordinary Hebbian plasticity reading those waves builds retinotopic maps
and even primes direction-selective circuits *before any visual input exists*.

**Codebase mapping — and why this is not hardcoding.** Add a phase-0 "development"
period before the behavioural lifetime (structurally the same hook `--bootcamp`
already provides): the body does not move and reward is off; instead, inject
correlated travelling-wave activity into the sensor pools — e.g. a bump of activation
sweeping `sL → sR` and back, at whatever rate matches the creature's own turning
dynamics. Plain Hebbian STDP then builds sensor→motor structure from the wave's
correlation statistics. Critically, **the correlation structure comes from the sensor
geometry, not from a designer-specified sensor→motor mapping** — you never say which
motor pool should win, you only supply structured pre-synaptic activity, exactly as
the retina does. Whether that clears your constraint bar is a judgement call worth
making explicitly before running it, but the biological precedent is strong and the
claim "competence emerges from network dynamics/plasticity" survives.

**Cost.** ~30 LOC (a second `run()` call with `reflex_scale=0`, motion frozen, and a
synthetic sensor driver replacing the food geometry). 2 runs. Cheap because the
two-phase lifetime plumbing from Task 7 already exists.
**Bonus:** this also gives the eye pool (Task 8's null result) a *routing hypothesis* —
a wave sweeping across the 20 photoreceptors would build topography that random
distance-based wiring cannot. Task 8's stated conclusion asked for exactly this.

---

#### D2 — Critical-period plasticity gating

**Source:** [Achille, Rovere & Soatto 2019, ICLR, "Critical learning periods in deep networks"](https://openreview.net/forum?id=BkeStsCcKQ);
[Critical learning periods emerge even in deep linear networks, ICLR 2024](https://openreview.net/pdf?id=Aq35gl2c1k).

**Mechanism.** Learning systems pass through an early window during which the
information structure acquired determines the final solution; degrading input during
that window causes permanent deficits that later training cannot repair. The
mechanistic account is that early high plasticity sets the connectivity skeleton and
later low plasticity refines it.

**Codebase mapping.** Anneal `A_PLUS`/`A_MINUS` across the lifetime (high for the
first ~1000 steps, decaying after), rather than holding them constant for all 6000
steps. ~8 lines and one new flag. Almost free.

**Interaction worth testing.** `--fep-punish` scrambles sensory input on failure. If
critical-period logic holds, punishment bursts landing in the *early* window would be
disproportionately damaging — which is a candidate explanation for Task 12's finding
that harsher punishment was strictly worse and Task 11's that softer was also worse
(a non-monotonic bracket is what you'd expect if the *timing* rather than the
intensity is the operative variable). Gating punishment off during the first N steps
is a 3-line experiment that tests this directly and is arguably the single
highest-information-per-minute run available.

**Cost.** ~8 LOC, 1-2 runs. Trivial.

---

### E. Meta-level: evolve the plasticity rule, not the weights

#### E1 — Evolved per-synapse Hebbian coefficients (ABCD rules) on random weights

**Source:** [Najarro & Risi 2020, NeurIPS, "Meta-Learning through Hebbian Plasticity in Random Networks"](https://arxiv.org/abs/2007.02686)
([code](https://github.com/enajx/HebbianMetaLearning)).

**Mechanism.** Do not evolve weights at all. Initialise every weight randomly at the
start of each lifetime, and evolve instead a small set of per-synapse Hebbian
coefficients `(A, B, C, D, η)` in the rule
`Δw = η(A·pre·post + B·pre + C·post + D)`. The network self-organises from random
init *within a single episode*. Their agents learn to walk from random weights in
under 100 timesteps and adapt to unseen morphological damage — with no reward signal
delivered to the network itself.

**Why this directly addresses a finding you already have.** Task 6 established that
evolving the weight vector for 100 generations against 4 fixed training seeds
produced pure overfitting: training fitness 8.4 → 14.75 while held-out food went
*down*. That is a symptom of an enormous genome (one gene per plastic synapse,
thousands of dimensions) memorising seed-specific quirks. Evolving a rule instead
collapses the genome to a handful of coefficients per *pool pair* — orders of
magnitude fewer dimensions, and a genome that cannot express "food is at (0.3, 0.7)
on seed 2" even in principle. **This is the most direct available fix for the
overfitting result, and it has not been tried.**

**Codebase mapping.** Genome changes from `(B, n_plastic)` weights to `(B, n_coeffs)`
rule parameters; `weight_plastic` is re-randomised each lifetime instead of
inherited. Broadcast coefficients per source/target pool pair (say 8 pools × 2 motor
targets × 5 coefficients ≈ 80 genes, vs. thousands now). Requires touching the GA
side (`clamp_genome`, `crossover`, `mutate`, `breed_next_generation`) and the
`run_lifetime` init — the largest single change on this list, but all of it is
mechanical. Dale's principle enforced by the same clamp as always.

**Cost.** ~100-150 LOC across sim + GA. 3-4 runs including a coefficient-range
bracket. ~90 min GPU. Highest cost, but highest ceiling, and it makes every other
rule on this list searchable rather than hand-tuned.

---

#### E2 — Cartesian genetic programming over plasticity rules

**Source:** [Jordan, Schmidt, Senn & Petrovici 2021, *eLife* 10:e66273, "Evolving interpretable plasticity for spiking networks"](https://elifesciences.org/articles/66273);
see also [Evolving-to-learn RL tasks with SNNs](https://arxiv.org/html/2202.12322).

**Mechanism.** Evolve the *symbolic expression* of the plasticity rule with genetic
programming over local biophysical variables, rather than tuning coefficients of a
fixed functional form. Recovered known rules and discovered novel ones — including
reward-modulated homeostatic terms that encourage exploration early in learning, and
optimistic reward baselines.

**Assessment.** Strictly more expressive than E1 and produces human-readable rules
you could write up. But the search is far more expensive and the reported successes
are single-neuron / small-network tasks, not embodied closed loops. **Do E1 first;
E2 is the follow-up if E1's fixed ABCD form turns out to be the binding constraint.**

---

### F. Predictive coding beyond the current FEP-punish version

#### F1 — Efference-copy self-supervision

**Source:** [Scherr, Stöckl & Maass 2022, NeurIPS, "Self-Supervised Learning Through Efference Copies"](https://papers.neurips.cc/paper_files/paper/2022/file/1d1cea122b9ec9f78acc21510659e500-Paper-Conference.pdf).

**Mechanism.** Use the agent's own motor command (efference copy) as the
self-supervision target: the network learns representations by predicting the sensory
consequences of its own actions. No labels, no reward — the action itself supplies
the training signal.

**Codebase mapping.** Designate a subset of interneurons as a prediction pool
receiving efference copy from `mL`/`mR` plus current sensors; train sensor→prediction
synapses by plain Hebbian toward next-step `L`/`R`; then use the prediction error as
the third factor for motor-pool synapses. This is the *principled* version of what
`--fep-punish` approximates crudely (punish = artificially destroy predictability;
this = actually measure it). ~60 LOC.

**Assessment.** Intellectually the strongest continuation of the Task 9 story, and it
would let you replace the environment-side hack with an in-brain quantity. But it is
a substantial build and depends on the network being able to learn a forward model at
all, which is unvalidated here. **Rank below A1/A2 for now; revisit if a
symmetry-breaking fix lands and the bottleneck moves.** See the
[predictive coding with SNNs survey (arXiv:2409.05386)](https://arxiv.org/abs/2409.05386)
for the current landscape.

---

## Part 3 — Ranked shortlist

Ranked by (plausibility of beating r = 0.080) × (1 / implementation cost).

| # | Candidate | Why it could beat 0.080 | LOC | Runs |
|---|---|---|---|---|
| **0** | **`asym` weight-space diagnostic** (§1.3) | Not a mechanism — a measurement. Tells you within one run whether *any* candidate differentiates the sL→mL vs sL→mR pathways, instead of inferring it from noisy behavioural r. Do this before anything else. | ~15 | 0 (piggyback) |
| **1** | **A1 — EH / reward-modulated Hebbian with double baseline** | Supplies the one term provably missing from every rule tried: a *signed* postsynaptic-fluctuation factor that can potentiate mL and depress mR for the same input. Published as sufficient to make a chaotic recurrent net a controller from scalar reward alone. Cheap, orthogonal to R-STDP and to `surprise`. | ~30 | 2 |
| **2** | **A2 — subtractive normalisation + fast heterosynaptic depression** | Miller & MacKay: multiplicative constraints *cannot* break symmetry, subtractive ones can — and the current homeostasis is multiplicative and 500 steps too slow. Smallest possible edit to the current best mechanism; the most likely single explanation for the universal 20-40% ceiling. Stacks with #1. | ~25 | 2 |
| **3** | **D2 + B2 — critical-period gating and event-gated behavioural-timescale eligibility** | Two ~10-20 line experiments that reuse existing machinery. D2 tests whether punishment *timing* (not intensity) explains the Task 11/12 bracket — the highest information-per-GPU-minute run available. B2 extends the already-implemented `elig` tensor to the actual behavioural loop length (~1000 steps) and side-steps the Task 10 interference by being event-gated. | ~28 | 2-3 |
| **4** | **E1 — evolve the plasticity rule, not the weights (Najarro & Risi)** | The direct fix for Task 6's overfitting result: collapses the genome from thousands of per-synapse genes to ~80 rule coefficients, which cannot memorise seed-specific layouts. Published as learning from random init *within one lifetime*, which is exactly the stated problem. Highest cost, highest ceiling; also turns every other rule here into something searchable. | ~150 | 3-4 |
| **5** | **C1 — DEP / differential extrinsic plasticity** | The only candidate whose published claim is literally "spontaneous symmetry breaking from brain-body-environment coupling", with emergent coordinated behaviour in minutes and zero reward. Completely orthogonal to every reward-based dead end in Tasks 4-12. Highest translation risk (rate→spiking derivatives), so ranked below the cheap fixes despite strong conceptual fit. | ~45 | 2-3 |

**Suggested execution order:** 0 → 2 → 1 → 3 → then 4 or 5 depending on whether the
`asym` diagnostic shows the pathways differentiating (if yes, the rule is working and
E1 lets you search it properly; if no, DEP's reward-free route is the better bet).

**Stacking note.** A2, A3 and A4 are substrate fixes, not competing rules — each can
sit under FEP, EH, or DEP. If A2 alone moves `asym` at all, run A2 + A1 next rather
than treating them as alternatives.

**What not to do again:** more punishment-parameter sweeps (bracketed on both sides,
Tasks 11-12); combining `surprise` with `--fep-punish` (Task 10); more raw sensory
channels without a routing hypothesis (Task 8) — D1 supplies the missing routing
hypothesis if the eye pool is revisited; more generations against a fixed 4-seed
training set (Task 6 — that lever produces overfitting, not competence).

---

## Sources

- [Miller & MacKay 1994, *Neural Computation* — The Role of Constraints in Hebbian Learning](https://direct.mit.edu/neco/article/6/1/100/5780/The-Role-of-Constraints-in-Hebbian-Learning)
- [Zenke & Gerstner 2017, *Phil. Trans. R. Soc. B* — Hebbian plasticity requires compensatory processes on multiple timescales](https://royalsocietypublishing.org/rstb/article/372/1715/20160259/23102/Hebbian-plasticity-requires-compensatory-processes)
- [Legenstein, Chase, Schwartz & Maass 2010, *J. Neurosci.* — A reward-modulated Hebbian learning rule can explain network reorganization in a brain control task](https://www.jneurosci.org/content/30/25/8400)
- [Hoerzer, Legenstein & Maass 2014, *Cerebral Cortex* — Emergence of complex computational structures from chaotic neural networks through reward-modulated Hebbian learning](https://academic.oup.com/cercor/article/24/3/677/392266)
- [Vogels, Sprekeler, Zenke, Clopath & Gerstner 2011, *Science* — Inhibitory plasticity balances excitation and inhibition](https://www.science.org/doi/10.1126/science.1211095)
- [Triesch 2005, ICANN — A gradient rule for the plasticity of a neuron's intrinsic excitability](https://link.springer.com/chapter/10.1007/11550822_11)
- [Lazar, Pipa & Triesch 2009, *Front. Comput. Neurosci.* — SORN: a self-organizing recurrent neural network](https://pubmed.ncbi.nlm.nih.gov/19893759/)
- [Aswolinskiy & Pipa 2015, *Front. Comput. Neurosci.* — RM-SORN: a reward-modulated self-organizing recurrent neural network](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00036/full)
- [Fiete & Seung 2006, *Phys. Rev. Lett.* — Gradient learning in spiking neural networks by dynamic perturbation of conductances](https://dx.doi.org/10.1103/PhysRevLett.97.048104)
- [Fiete, Fee & Seung 2007, *J. Neurophysiol.* — Model of birdsong learning based on gradient estimation by dynamic perturbation](https://journals.physiology.org/doi/full/10.1152/jn.01311.2006)
- [Hiratani, Mehta, Lillicrap & Latham 2022, NeurIPS — On the stability and scalability of node perturbation learning](https://proceedings.neurips.cc/paper_files/paper/2022/file/cf38eb1549024cce4b3d2c1bb87a6c27-Paper-Conference.pdf)
- [Bittner, Milstein et al. 2017, *Science* — Behavioral time scale synaptic plasticity underlies CA1 place fields](https://www.science.org/doi/10.1126/science.aan3846)
- [Gerstner, Lehmann, Liakoni, Corneil & Brea 2018, *Front. Neural Circuits* — Eligibility traces and plasticity on behavioral time scales](https://arxiv.org/abs/1801.05219)
- [Cone & Shouval 2025, *Nat. Commun.* — A simple model for BTSP gives content-addressable memory with one-shot learning](https://www.nature.com/articles/s41467-024-55563-6)
- [Der & Martius 2015, *PNAS* — Novel plasticity rule can explain the development of sensorimotor intelligence (DEP)](https://www.pnas.org/doi/10.1073/pnas.1508400112) · [arXiv](https://arxiv.org/abs/1505.00835)
- [Martius, Der & Ay 2013, *PLOS ONE* — Information driven self-organization of complex robotic behaviors](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0063400)
- [Porr & Wörgötter — Isotropic sequence order learning (ISO/ICO)](https://www.berndporr.me.uk/isolearn/isolearning.pdf)
- [Kolodziejski, Porr & Wörgötter 2009, *Biol. Cybern.* — Mathematical properties of neuronal TD-rules and differential Hebbian learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC2798052/)
- [Ge et al. 2021, *Science* — Retinal waves prime visual motion detection by simulating future optic flow](https://www.science.org/doi/10.1126/science.abd0830)
- [Chandrasekaran/Cang et al., *Neuron* — Retinotopic map refinement requires spontaneous retinal waves during a critical period](https://www.cell.com/neuron/fulltext/S0896-6273(03)00790-6)
- [Hebbian instruction of axonal connectivity by endogenous correlated spontaneous activity, *Science* 2023](https://www.science.org/doi/10.1126/science.adh7814)
- [Achille, Rovere & Soatto 2019, ICLR — Critical learning periods in deep networks](https://openreview.net/forum?id=BkeStsCcKQ)
- [Critical learning periods emerge even in deep linear networks, ICLR 2024](https://openreview.net/pdf?id=Aq35gl2c1k)
- [Najarro & Risi 2020, NeurIPS — Meta-learning through Hebbian plasticity in random networks](https://arxiv.org/abs/2007.02686) · [code](https://github.com/enajx/HebbianMetaLearning)
- [Jordan, Schmidt, Senn & Petrovici 2021, *eLife* — Evolving interpretable plasticity for spiking networks](https://elifesciences.org/articles/66273)
- [Scherr, Stöckl & Maass 2022, NeurIPS — Self-supervised learning through efference copies](https://papers.neurips.cc/paper_files/paper/2022/file/1d1cea122b9ec9f78acc21510659e500-Paper-Conference.pdf)
- [Predictive coding with spiking neural networks: a survey, arXiv:2409.05386](https://arxiv.org/abs/2409.05386)
- [Pfister & Gerstner triplet STDP / BCM generalization, *PNAS* 2011](https://www.pnas.org/doi/10.1073/pnas.1105933108)
- [Kagan et al. 2022, *Neuron* — In vitro neurons learn when embodied in a simulated game-world (DishBrain)](https://www.cell.com/neuron/fulltext/S0896-6273(22)00806-6)
