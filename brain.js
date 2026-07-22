/* ============================================================================
 * brain.js — shared, fully-specced organoid brain as an instantiable module.
 *
 * This is the SAME computational-neuroscience machinery as index.html
 * (Izhikevich spiking neurons, Dale's principle, STDP, homeostatic synaptic
 * scaling, metabolic energy/fatigue, distance-dependent wiring), but wrapped
 * in a class with NO module-level globals — so you can run many brains at once
 * (heats, populations, side-by-side comparisons) instead of the single
 * one-brain-in-scope model the original page uses.
 *
 * It is deliberately SMALL by default (N ~ 120) but keeps EVERY feature. Cost
 * scales with N * K per tick; shrink N, keep the biology.
 *
 * Usage (browser):   <script src="brain.js"></script>  then  new Brain(genome)
 * Usage (node):      const {Brain, randomGenome, mutate} = require('./brain.js')
 *
 * Contract:
 *   const b = new Brain(genome);          // genome from randomGenome()/mutate()
 *   const [mL, mR] = b.step([L,R,wall,foodClose,hunger,tired]);  // one tick
 *   b.reward(0.85);                        // dopamine pulse (e.g. on feeding)
 *   b.plastic = true;                      // enable STDP (learning) — default on
 * ========================================================================== */
(function (root) {
  'use strict';

  // ---- seeded RNG (mulberry32) so a genome reproduces the same structure ----
  function makeRng(seed) {
    let s = seed >>> 0;
    return function () {
      s |= 0; s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Izhikevich subtype constants
  const RS = 0, IB = 1, FS = 2, LTS = 3;

  // STDP / plasticity constants (identical to index.html)
  const TRACE_DECAY = 0.96, A_PLUS = 0.008, A_MINUS = 0.010, W_MAX = 2.5;

  // ---- genome helpers --------------------------------------------------------
  // A genome fully determines a brain: a structural seed (wiring/positions/
  // types) plus continuous, heritable parameters. Breeding keeps the seed
  // (structure is inherited) and perturbs the continuous genes; rarely it
  // reseeds to explore new architectures.
  function randomGenome(rand) {
    rand = rand || Math.random;
    return {
      seed: (rand() * 4294967296) >>> 0,
      N: 120,
      fi: 0.15 + rand() * 0.15,        // inhibitory fraction 0.15–0.30
      K: 6 + Math.floor(rand() * 5),   // connections/neuron 6–10
      longRangeP: rand() * 0.12,       // fraction of long-range synapses
      reflexL: 0.4 + rand() * 0.5,     // innate sL->mL reflex
      reflexR: 0.4 + rand() * 0.5,     // innate sR->mR reflex
      reflexW: 0.2 + rand() * 0.4,     // innate wall reflex
      sensorGain: 1.0 + rand() * 1.6,  // body->brain input gain
      motorGain: 0.7 + rand() * 1.2,   // brain->body output gain
      homeoTarget: 0.015 + rand() * 0.02,
      excit: 0.8 + rand() * 0.5,       // global excitability
      selfOrg: 0.0                     // developmental migration (off by default)
    };
  }

  function clampG(g) {
    g.fi = Math.min(0.4, Math.max(0.05, g.fi));
    g.K = Math.min(12, Math.max(4, Math.round(g.K)));
    g.longRangeP = Math.min(0.3, Math.max(0, g.longRangeP));
    g.reflexL = Math.min(1.4, Math.max(0, g.reflexL));
    g.reflexR = Math.min(1.4, Math.max(0, g.reflexR));
    g.reflexW = Math.min(1.0, Math.max(0, g.reflexW));
    g.sensorGain = Math.min(3.0, Math.max(0.3, g.sensorGain));
    g.motorGain = Math.min(2.5, Math.max(0.3, g.motorGain));
    g.homeoTarget = Math.min(0.06, Math.max(0.005, g.homeoTarget));
    g.excit = Math.min(1.6, Math.max(0.5, g.excit));
    return g;
  }

  // Breed: copy parent genome, perturb genes. rate ~ mutation strength.
  function mutate(genome, rate, rand) {
    rand = rand || Math.random;
    rate = rate == null ? 0.15 : rate;
    const g = Object.assign({}, genome);
    const j = (v, scale) => v + (rand() * 2 - 1) * rate * scale;
    g.fi = j(g.fi, 0.15);
    g.K = j(g.K, 3);
    g.longRangeP = j(g.longRangeP, 0.08);
    g.reflexL = j(g.reflexL, 0.5);
    g.reflexR = j(g.reflexR, 0.5);
    g.reflexW = j(g.reflexW, 0.4);
    g.sensorGain = j(g.sensorGain, 0.8);
    g.motorGain = j(g.motorGain, 0.6);
    g.homeoTarget = j(g.homeoTarget, 0.01);
    g.excit = j(g.excit, 0.3);
    // 5% chance to explore a new architecture (reseed structure)
    if (rand() < 0.05) g.seed = (rand() * 4294967296) >>> 0;
    return clampG(g);
  }

  // ---- Brain -----------------------------------------------------------------
  class Brain {
    constructor(genome) {
      this.genome = clampG(Object.assign(randomGenome(), genome || {}));
      this.rng = makeRng(this.genome.seed);
      this.plastic = true;    // STDP on (this is the whole point vs. the stub)
      this.dale = true;       // inhibition functional
      this.dopamine = 0;
      this.stepCount = 0;
      this.noise = 0.03;
      // Drive tuning — small networks have a thinner recurrent "sea" than the
      // 600-neuron original, so sensory injection and innate reflexes are scaled
      // up to keep motor pools reaching threshold. Overridable via genome.
      this.injMult = this.genome.injMult || 22.0;
      this.reflexScale = this.genome.reflexScale || 3.0;
      this._build();
    }

    // gaussian via the instance rng
    _gauss() {
      let u = 0, v = 0;
      while (!u) u = this.rng();
      while (!v) v = this.rng();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283 * v);
    }

    _build() {
      const g = this.genome, R = this.rng;
      const N = this.N = g.N | 0;

      // Pool layout scales with N. 6 sensory pools + 2 motor pools, the rest
      // are interneurons. (index.html hard-codes indices for N>=190; here we
      // allocate proportionally so small brains still have every pool.)
      const sp = Math.max(4, Math.round(N * 0.06));   // neurons per sensory pool
      const mp = Math.max(6, Math.round(N * 0.11));   // neurons per motor pool
      let c = 0;
      const P = {};
      ['sL', 'sR', 'sW', 'sD', 'sH', 'sT'].forEach(k => { P[k] = [c, c + sp]; c += sp; });
      P.mL = [c, c + mp]; c += mp;
      P.mR = [c, c + mp]; c += mp;
      this.pool = P;
      this.interStart = c; // interneurons occupy [c, N)

      // typed arrays
      this.px = new Float32Array(N); this.py = new Float32Array(N); this.pz = new Float32Array(N);
      this.types = new Int8Array(N); this.subtypes = new Int8Array(N);
      this.v = new Float32Array(N); this.u = new Float32Array(N);
      this.a = new Float32Array(N); this.b = new Float32Array(N);
      this.c = new Float32Array(N); this.d = new Float32Array(N);
      this.energy = new Float32Array(N); this.fatigue = new Float32Array(N);
      this.tracePre = new Float32Array(N); this.tracePost = new Float32Array(N);
      this.smooth = new Float32Array(N); this.firing = new Float32Array(N);
      this.sensorInj = new Float32Array(N);
      this.spikeCount = new Int32Array(N);

      // positions on a unit ball, types (E/I), energy
      for (let i = 0; i < N; i++) {
        let x, y, z, r2;
        do { x = R() * 2 - 1; y = R() * 2 - 1; z = R() * 2 - 1; r2 = x * x + y * y + z * z; } while (r2 > 1);
        this.px[i] = x; this.py[i] = y; this.pz[i] = z;
        this.types[i] = R() < g.fi ? -1 : 1;
        this.energy[i] = 1.0;
      }
      // sensory + motor pools are excitatory
      for (const k in P) for (let i = P[k][0]; i < P[k][1]; i++) this.types[i] = 1;

      // Izhikevich subtype params with small per-cell jitter
      for (let i = 0; i < N; i++) {
        let ba, bb, bc, bd;
        if (this.types[i] > 0) {
          if (R() < 0.85) { this.subtypes[i] = RS; ba = 0.02; bb = 0.2; bc = -65; bd = 8; }
          else { this.subtypes[i] = IB; ba = 0.02; bb = 0.2; bc = -50; bd = 2; }
        } else {
          if (R() < 0.80) { this.subtypes[i] = FS; ba = 0.1; bb = 0.2; bc = -65; bd = 2; }
          else { this.subtypes[i] = LTS; ba = 0.02; bb = 0.25; bc = -65; bd = 2; }
        }
        const rv = () => (R() * 2 - 1);
        this.a[i] = ba * (1 + rv() * 0.08);
        this.b[i] = bb * (1 + rv() * 0.08);
        this.c[i] = bc + rv() * 2.0;
        this.d[i] = bd * (1 + rv() * 0.08);
        this.v[i] = this.c[i]; this.u[i] = this.b[i] * this.v[i];
      }

      // distance-dependent wiring (K nearest-ish, prob ~ exp(-d^2), long-range)
      this.inc = [];
      for (let i = 0; i < N; i++) this.inc.push([]);
      const decay = 0.6;
      for (let jn = 0; jn < N; jn++) {
        const ds = [];
        for (let k = 0; k < N; k++) {
          if (k === jn) continue;
          const dx = this.px[jn] - this.px[k], dy = this.py[jn] - this.py[k], dz = this.pz[jn] - this.pz[k];
          ds.push([dx * dx + dy * dy + dz * dz, k]);
        }
        ds.sort((p, q) => p[0] - q[0]);
        let found = 0, idx = 0;
        while (found < g.K && idx < ds.length) {
          const nb = ds[idx][1], dist2 = ds[idx][0];
          let prob = Math.exp(-dist2 / decay);
          if (R() < g.longRangeP) prob = 1.0;
          if (R() < prob) {
            this.inc[jn].push({ pre: nb, weight: Math.abs(this._gauss()) * this.types[nb] * 0.8, activity: 0.5, age: 0, reflex: false });
            found++;
          }
          idx++;
        }
      }
      this._norm();

      // innate reflex synapses (the only hand-wired paths; everything else learns).
      // Stride adapts to pool size so small brains still get enough reflex drive
      // to steer — with a big fixed stride, a 7-neuron sensory pool would wire
      // only 2-3 reflexes and the motors would never fire.
      const stride = Math.max(1, Math.floor(sp / 6));
      const rs = this.reflexScale;
      const link = (spk, mpk, wt) => {
        for (let t = P[mpk][0]; t < P[mpk][1]; t++)
          for (let s = P[spk][0]; s < P[spk][1]; s += stride)
            this.inc[t].push({ pre: s, weight: wt * rs * this.types[s], activity: 1.0, age: 500, reflex: true });
      };
      link('sL', 'mL', g.reflexL);
      link('sR', 'mR', g.reflexR);
      link('sW', 'mL', g.reflexW); link('sW', 'mR', g.reflexW);

      this._rebuildOut();
    }

    _norm() {
      for (let i = 0; i < this.N; i++) {
        let s = 0; const syn = this.inc[i];
        for (const e of syn) s += Math.abs(e.weight);
        if (s > 0) { const gn = 1.4 / s; for (const e of syn) e.weight *= gn; }
      }
    }

    // outgoing adjacency for O(1) STDP depression lookups
    _rebuildOut() {
      this.out = [];
      for (let i = 0; i < this.N; i++) this.out.push([]);
      for (let i = 0; i < this.N; i++)
        for (const syn of this.inc[i])
          if (!syn.reflex) this.out[syn.pre].push({ post: i, syn: syn });
      this.outDirty = false;
    }

    reward(amt) { this.dopamine = Math.min(2.0, this.dopamine + (amt == null ? 0.85 : amt)); }

    poolMean(k) {
      const r = this.pool[k]; let s = 0;
      for (let i = r[0]; i < r[1]; i++) s += this.smooth[i];
      return s / (r[1] - r[0]);
    }

    // Inject a sensor vector [L, R, wall, foodClose, hunger, tired] into the
    // sensory pools, advance the spiking dynamics one tick, return [mL, mR].
    step(sensors) {
      const g = this.genome, N = this.N, P = this.pool;
      const Sg = g.sensorGain, sIn = this.sensorInj;
      sIn.fill(0);
      const inj = (k, val) => { const r = P[k]; const w = val * Sg * this.injMult; for (let i = r[0]; i < r[1]; i++) sIn[i] = w; };
      inj('sL', sensors[0] || 0); inj('sR', sensors[1] || 0); inj('sW', sensors[2] || 0);
      inj('sD', sensors[3] || 0); inj('sH', sensors[4] || 0); inj('sT', sensors[5] || 0);

      this._stepBrain();

      const mL = this.poolMean('mL'), mR = this.poolMean('mR');
      return [mL, mR];
    }

    _stepBrain() {
      const N = this.N;
      this.stepCount++;
      this.dopamine *= 0.985;
      const noiseScale = this.noise;
      const excit = this.genome.excit;
      const spiked = [];

      const tp = this.tracePre, tq = this.tracePost;
      for (let i = 0; i < N; i++) { tp[i] *= TRACE_DECAY; tq[i] *= TRACE_DECAY; }

      for (let i = 0; i < N; i++) {
        this.energy[i] += (1.0 - this.energy[i]) * 0.001;
        this.fatigue[i] *= 0.98;
        const exc = this.energy[i] * (1.0 - this.fatigue[i] * 0.5);

        let Isyn = 0; const syn = this.inc[i];
        for (let m = 0; m < syn.length; m++) {
          const s = syn[m];
          Isyn += (this.dale ? s.weight : Math.abs(s.weight)) * this.smooth[s.pre];
          s.age += 1;
        }
        let Ieff = Isyn * exc;
        if (this.sensorInj[i]) Ieff += this.sensorInj[i] * exc;

        let Inoise = this._gauss() * noiseScale;
        if (this.rng() < 0.002) Inoise += 15.0;
        let I = Ieff * excit + Inoise;

        let vi = this.v[i], ui = this.u[i];
        const ai = this.a[i], bi = this.b[i];
        if (this.fatigue[i] > 0.6) I *= (1.0 - (this.fatigue[i] - 0.6) * 0.5);

        for (let st = 0; st < 2; st++) {
          vi += 0.5 * (0.04 * vi * vi + 5 * vi + 140 - ui + I);
          ui += 0.5 * ai * (bi * vi - ui);
        }
        if (vi > 35) vi = 35; if (vi < -90) vi = -90;
        this.v[i] = vi; this.u[i] = ui;

        let sp = 0;
        if (vi >= 30) {
          sp = 1;
          this.v[i] = this.c[i]; this.u[i] += this.d[i];
          this.energy[i] = Math.max(0, this.energy[i] - 0.002);
          this.fatigue[i] = Math.min(1.0, this.fatigue[i] + 0.05);
          this.spikeCount[i]++;
          spiked.push(i);
        }
        this.smooth[i] = this.smooth[i] * 0.8 + sp * 0.2;
        this.firing[i] = this.firing[i] * 0.99 + sp * 0.01;
      }

      // STDP (reward-modulated)
      if (this.plastic && spiked.length > 0) {
        if (this.outDirty) this._rebuildOut();
        // Reward-gated plasticity (DishBrain premise): with little recent dopamine
        // STDP is nearly frozen, so it doesn't erode the innate reflex pathways
        // between meals; a reward pulse opens a strong learning window.
        const dopa = 0.03 + this.dopamine * 5.0;
        for (let s = 0; s < spiked.length; s++) {
          const idx = spiked[s];
          tp[idx] += 1.0; tq[idx] += 1.0;
          const syn = this.inc[idx];
          for (let m = 0; m < syn.length; m++) {
            const sy = syn[m]; if (sy.reflex) continue;
            const pre = sy.pre;
            const ef = Math.min(this.energy[pre], this.energy[idx]);
            const dw = A_PLUS * tp[pre] * ef * dopa;
            if (this.types[pre] > 0) { sy.weight += dw; if (sy.weight > W_MAX) sy.weight = W_MAX; if (sy.weight < 0) sy.weight = 0; }
            else { sy.weight -= dw; if (sy.weight < -W_MAX) sy.weight = -W_MAX; if (sy.weight > 0) sy.weight = 0; }
            sy.activity = sy.activity * 0.95 + 0.05;
          }
          const og = this.out[idx];
          for (let m = 0; m < og.length; m++) {
            const tgt = og[m].post, sy = og[m].syn;
            const ef = Math.min(this.energy[idx], this.energy[tgt]);
            const dw = A_MINUS * tq[tgt] * ef * dopa;
            if (this.types[idx] > 0) { sy.weight -= dw; if (sy.weight < 0) sy.weight = 0; }
            else { sy.weight += dw; if (sy.weight > 0) sy.weight = 0; }
          }
        }
      }

      // Homeostatic synaptic scaling
      if (this.stepCount % 500 === 0) {
        const target = this.genome.homeoTarget;
        for (let i = 0; i < N; i++) {
          const rate = this.firing[i]; const syn = this.inc[i];
          for (let m = 0; m < syn.length; m++) {
            const sy = syn[m]; if (sy.reflex) continue;
            let f = 1.0;
            if (rate < target) f = 1.05; else if (rate > target) f = 0.95;
            sy.weight *= f;
            if (this.types[sy.pre] > 0) { if (sy.weight > W_MAX) sy.weight = W_MAX; if (sy.weight < 0) sy.weight = 0; }
            else { if (sy.weight < -W_MAX) sy.weight = -W_MAX; if (sy.weight > 0) sy.weight = 0; }
          }
        }
        if (this.rng() < 0.2) this._norm();
      }
    }
  }

  const api = { Brain, randomGenome, mutate, makeRng };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else { root.Brain = Brain; root.BrainKit = api; }
})(typeof window !== 'undefined' ? window : globalThis);
