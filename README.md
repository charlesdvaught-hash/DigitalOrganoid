# 🧫🐛 Organoid Creature — an embodied closed-loop brain sim

A single-file, dependency-free HTML simulation. A tiny creature forages for food in an arena; its "eyes" feed sensor neurons in a spiking neural network, and motor neurons steer it. The network is not a controller bolted onto the creature — the organoid *is* the brain, and it runs a full closed perception–action loop. Open the file in any modern browser; click the arena to drop food.

The left canvas is the arena, the middle canvas is the live 3D network, and the right panel holds the controls, a simulated multi-electrode-array (MEA) readout, and vitals.

## What makes it realistic

The network runs on genuine computational-neuroscience machinery rather than hand-tuned game logic:

**Izhikevich spiking neurons.** Every cell integrates the Izhikevich model (`v` membrane potential, `u` recovery variable), the standard efficient model that reproduces real cortical firing patterns. Neurons are typed by biophysical subtype — regular-spiking and intrinsically-bursting excitatory cells, fast-spiking and low-threshold-spiking inhibitory cells — each with its own `a/b/c/d` parameters, plus small cell-to-cell parameter jitter so no two neurons are identical.

**Dale's principle.** A neuron is either excitatory or inhibitory and all its outgoing synapses share that sign. The Dale toggle actually enforces this: with it on, inhibitory cells subtract; with it off, inhibition becomes non-functional and you can watch the network destabilize into runaway excitation.

**Spike-timing-dependent plasticity (STDP).** Synapses potentiate and depress based on pre/post spike traces, bounded by a maximum weight — the network learns from its own activity, not from a global error signal.

**Homeostatic plasticity.** Periodic synaptic scaling nudges each neuron toward a target firing rate, and weight normalization keeps total input in range — the same stabilizing mechanisms that keep real networks from saturating or going silent.

**Metabolic constraints.** Each neuron has energy and fatigue. Spiking costs energy and accrues fatigue, which throttles excitability and recovers over time. Firing is not free, so activity has to be economical — a real pressure that shapes the network.

**Developmental self-organization.** Neurons live in a 3D ball and physically migrate under activity correlation, cell-type similarity, and a diffusing morphogen field, with short-range repulsion preventing collapse. Synapses are wired by distance-dependent probability with a tunable fraction of long-range connections. Over time the network prunes weak/unused synapses and grows new ones toward correlated partners — activity-dependent structural plasticity, the coarse shape of how real circuits refine themselves.

**Simulated MEA readout.** An 8×8 electrode grid bins spikes spatially and reports spike rate, network bursts, and a synchrony index — mirroring how lab organoids are actually recorded, including the characteristic synchronized bursting of immature cultures.

**Preset conditions.** The modifier checkboxes (Cortical Org, DishBrain Learning, Bursting Culture, High Self-Org, Metabolic Stress) map to recognizable regimes from the organoid and cultured-network literature, including the DishBrain-style reward-modulated learning setup.

## Be clear about the simplifications

This is a toy model, not a scientific instrument. The realism is structural, not quantitative:

- **Scale.** Hundreds to a few thousand neurons versus ~86 billion in a human brain (or millions in a real organoid). Connectivity per neuron is a small handful, not thousands.
- **Time is not calibrated.** Steps are simulation ticks tied to the animation frame, not milliseconds; the two Izhikevich sub-steps per tick are for stability, not biological timing. "Hz" on the MEA is a whole-network proxy, not per-electrode physiology.
- **No real neurochemistry.** "Dopamine," "energy," and "morphogen" are single scalars standing in for entire systems. There are no distinct neurotransmitters, receptor kinetics, dendritic computation, glia, or ion channels.
- **Reward is scripted.** Learning is driven by a hand-placed dopamine signal on feeding, not by anything the network discovers on its own.
- **Sensing and motor mapping are hard-coded.** Two directional food sensors plus a wall sensor feed fixed neuron pools, and two motor pools map directly to turn/speed. The body is a triangle with no physics beyond position and heading.
- **Development is compressed and stylized.** Migration forces, pruning, and growth are plausible-looking heuristics, not a model of real morphogenesis.
- **The 3D layout is illustrative.** Positions drive connection probability and the visualization, but the ball geometry is not anatomy.

Treat it as a sketch that captures the *flavor* of embodied neural dynamics — spikes, plasticity, homeostasis, metabolism, and self-organization interacting in a loop — while abstracting away almost all of the biology.

## What it could be used for

**Teaching.** It makes several hard-to-visualize ideas tangible at once: how spiking dynamics, inhibition, plasticity, and homeostasis interact, why unconstrained excitation is bad, and what a closed sensorimotor loop actually looks like. Toggling Dale's principle off and watching the network fall apart is a memorable lesson on its own.

**Intuition-building for researchers.** A fast sandbox for developing gut feel about parameter regimes — bursting versus asynchronous activity, the effect of connectivity and inhibition fraction, or how metabolic stress interacts with learning — before committing to a heavyweight simulator.

**Communicating organoid / "biological computing" concepts.** The MEA panel and DishBrain-style preset give a concrete, honest-about-its-limits illustration of what people mean by teaching a dish of neurons to do a task.

**A base to extend.** The closed loop is the valuable part. It's a natural starting point for experiments in embodied learning, neuromodulation schemes, curriculum design, or comparing learning rules — all in a system small enough to watch neuron-by-neuron.

## Ways to gamify it

The survival/foraging loop is already a game skeleton. Directions to push it:

**Survival & progression.** Score is already survival time and food eaten. Add hazards (toxins that drain energy, or "lesion" regions that kill neurons on contact), moving or fleeing food, day/night cycles, and difficulty that ramps as the creature survives longer.

**Breeding / evolution.** Let the best-surviving brains seed the next generation with mutated parameters and inherited wiring. Run a population, show a lineage, and let the player select for traits — turning the sim into an artificial-life breeder.

**"Train your dish" challenge.** Frame it explicitly as the DishBrain premise: give the player only the reward signal and the sliders, and score them on how fast they can teach the network a task (reach a target, avoid a wall, follow a light). Leaderboard on time-to-learn.

**Sandbox / god mode.** Let the player paint stimulation onto neurons, sever connections, inject dopamine, or lesion regions and watch behavior degrade and recover — a neural "SimCity" where the map is a brain.

**Puzzle mode.** Present a fixed brain and a task; the player must find slider/modifier settings that make it succeed. Each level locks certain controls, teaching one mechanism at a time (inhibition, then plasticity, then metabolism…).

**Collection & identity.** Give brains names, save/share seeds, and surface emergent "personalities" (cautious wall-hugger, greedy darter) from the stats. Snapshots of a favorite network's 3D structure make natural collectibles.

**Multiplayer / co-op.** Two creatures in one arena competing for food, or a shared brain that two players stimulate from opposite sides.

The honest framing to keep throughout: the player is shaping *conditions and rewards*, not scripting behavior — the fun is that the creature's competence is emergent, and sometimes it just refuses to thrive.

---

*Single file, no build step, no dependencies. Everything runs client-side in `organoid_creature1-1.html`.*
