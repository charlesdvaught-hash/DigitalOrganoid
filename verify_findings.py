"""Verification script: reproduces key claims from the findings report using the actual code."""
import json, numpy as np
from collections import defaultdict
from organoid_simulator.model import build_recurrent_weights, scale_spectral_radius, run_simulation, run_homeostasis, compute_spectral_radius
from organoid_simulator.metrics import compute_memory_capacity, solve_ridge_multi

out = {}

# --- 1. Aggregate the real sweep data by (N,K) at rho=0.95 ---
data = json.load(open('sweep_results.json'))
data += json.load(open('results_2e5.json'))
agg = defaultdict(lambda: defaultdict(list))
for r in data:
    if r.get('status') != 'success': continue
    if abs(r['rho'] - 0.95) > 1e-6: continue
    key = (r['N'], r['K'])
    agg[key]['mc'].append(r['total_mc'])
    agg[key]['br'].append(r['branching_ratio'])
    agg[key]['exp'].append(r['avalanche_exponent'])
rows = []
for (N,K) in sorted(agg):
    v = agg[(N,K)]
    rows.append(dict(N=N,K=K,n_seeds=len(v['mc']),
        mc_mean=round(float(np.mean(v['mc'])),3), mc_std=round(float(np.std(v['mc'])),3),
        br_mean=round(float(np.mean(v['br'])),3), exp_mean=round(float(np.mean(v['exp'])),3)))
out['sweep_aggregate'] = rows

# --- 2. Shift-register control: verifies the ridge solver can recover MC=10 exactly ---
# A perfect 10-tap linear delay line should yield total MC ~= 10.
np.random.seed(0)
T = 3000; N_sr = 10
u = np.random.uniform(-1,1,T).astype(np.float32)
# states = last 10 inputs (perfect shift register)
states = np.zeros((T,N_sr),dtype=np.float32)
for t in range(T):
    for d in range(N_sr):
        states[t,d] = u[t-d-1] if t-d-1>=0 else 0.0
mc_sr,_ = compute_memory_capacity(states,u,k_max=50,alpha=1e-8)
out['shift_register_control'] = dict(expected="~10.0 (10-tap delay line)", measured_mc=round(mc_sr,4))

# --- 3. Live reproduction: build a real N=300 reservoir and measure MC from scratch ---
def measure(N,K,rho,seed):
    W,ids = build_recurrent_weights(N,K,topology='random',seed=seed)
    W = scale_spectral_radius(W,rho)
    rho_meas = compute_spectral_radius(W)
    W_in,_ = run_homeostasis(W,target_mean=0.1,a=0.3)
    np.random.seed(seed+999)
    u = np.random.uniform(-1,1,5000).astype(np.float32)
    st = run_simulation(W,W_in,u,a=0.3)
    st = st[500:]; uu = u[500:]
    mc,_ = compute_memory_capacity(st,uu,k_max=50,alpha=1e-4)
    return round(mc,3), round(float(rho_meas),3)

live = []
for N in [100,300,1000]:
    mc,rm = measure(N,50,0.95,1)
    live.append(dict(N=N,K=50,rho_target=0.95,rho_measured=rm,mc_live=mc))
out['live_reproduction'] = live

# --- 4. rho sweep at N=300 to test 'MC peaks near edge of chaos' claim ---
rho_sweep=[]
for rho in [0.8,0.95,1.05]:
    mc,rm = measure(300,50,rho,1)
    rho_sweep.append(dict(rho=rho,mc=mc))
out['rho_sweep_N300'] = rho_sweep

print(json.dumps(out,indent=2))
