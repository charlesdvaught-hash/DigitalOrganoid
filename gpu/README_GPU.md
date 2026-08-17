# DigitalOrganoid GPU Engine & Flocke Enhancements

This directory contains the GPU-accelerated PyTorch evolutionary engine for DigitalOrganoid, featuring the 4 key Flocke-inspired architectural mechanisms:

1. **Innate CPG Prior (`--cpg`)**:
   - 2–4 neuron oscillator pool driving baseline motor rhythms (`mL` and `mR`) so the creature maintains active exploration even when recurrent network firing is low.

2. **Cerebellar Predictive Forward Model (`--cerebellum`)**:
   - 2-layer predictive filter estimating sensory state 1 step ahead.
   - Prediction error magnitude guides motor plasticity and feeds arousal signals.

3. **Multi-Channel Neuromodulation (`--multichannel-neuromod`)**:
   - Expands dopamine to 4 channels:
     - **Dopamine (DA)**: reward on feeding.
     - **Norepinephrine (NE)**: arousal / exploration driven by prediction error.
     - **Serotonin (5-HT)**: satiety / motor turn dampening.
     - **Acetylcholine (ACh)**: sensory attention / gain scaling.

4. **Bilateral Symmetric Initialization (`--bilateral-symmetric`)**:
   - Enforces sagittal-plane ($x \to -x$) mirrored initialization on sensorimotor pathways, eliminating directional seed variance.

---

## How to Run locally on Windows GPU

Double-click `run.bat` or execute in Command Prompt / PowerShell:

```cmd
run.bat
```

### Features of `run.bat`:
- **Auto-Detects CUDA GPU**: Uses `--device cuda` if NVIDIA PyTorch is installed, otherwise falls back gracefully to `--device cpu`.
- **Thorough Error Reporting**: If any error occurs, the window pauses and reports the exact error code and traceback so you can easily diagnose it.
- **Auto-Closes on Success**: When the benchmark completes without errors, it prints the summary and automatically closes after 3 seconds.

### Manual Command Example:

```cmd
python gpu_evolve.py --pop 64 --gens 60 --device cuda --cpg --cerebellum --multichannel-neuromod --bilateral-symmetric
```
