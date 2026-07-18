"""
Head-to-head: digital organoid reservoir  vs  small Transformer, on NARMA-10.

Fair comparison is reported as a Pareto trade-off (error vs. params vs. wall-time),
NOT a single winner. The reservoir sweeps size N; the transformer sweeps width/depth.

Reservoir  = scale-free + Dale's-law sparse recurrent net, non-negative activation
             (functional inhibition), fixed weights, trained linear (ridge) readout.
Transformer = small causal decoder over the input window, trained with Adam.

Runs on CPU for the reservoir; uses CUDA (your 5070) for the transformer if available.

Install:  pip install numpy scipy torch
Run:      python organoid_vs_transformer.py
"""
import os, time, argparse
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(_v,"4")
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------------------------------------------------------- task: NARMA-10
def narma10(T, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, T+50).astype(np.float64)   # standard NARMA input range
    y = np.zeros_like(u)
    for t in range(10, len(u)-1):
        y[t+1] = (0.3*y[t] + 0.05*y[t]*np.sum(y[t-9:t+1]) + 1.5*u[t-9]*u[t] + 0.1)
    return u[50:].astype(np.float32), y[50:].astype(np.float32)

def nrmse(pred, true):
    return float(np.sqrt(np.mean((pred-true)**2) / (np.var(true)+1e-12)))

# ------------------------------------------------- organoid reservoir (CPU, sparse)
def build_reservoir(N, K, rho, seed):
    rng = np.random.default_rng(seed)
    ids = np.ones(N, np.float32); ids[rng.choice(N, N-int(round(.8*N)), replace=False)] = -1
    # scale-free out-targets via preferential attachment
    indeg = np.ones(N); r=[]; c=[]; d=[]
    for j in range(N):
        p = indeg.copy(); p[j]=0; p/=p.sum()
        t = rng.choice(N, min(K,N-1), replace=False, p=p); indeg[t]+=1
        r+=list(t); c+=[j]*len(t); d+=list(np.abs(rng.standard_normal(len(t))).astype(np.float32)*ids[j])
    W = sp.csr_matrix((d,(r,c)), shape=(N,N), dtype=np.float32)
    ev = np.abs(spla.eigs(W.astype(np.float64), k=1, which='LM', return_eigenvectors=False, maxiter=500, tol=1e-4)[0])
    return (W*(rho/ev)).astype(np.float32), ids

def run_reservoir(W, u, gain, leak=0.3, act='tanh'):
    N=W.shape[0]; x=np.zeros(N,np.float32); S=np.zeros((len(u),N),np.float32)
    for t in range(len(u)):
        d = W.dot(x) + gain*u[t]
        f = np.tanh(d) if act=='tanh' else np.maximum(0.0, np.tanh(d))  # 'relu' = functional Dale's law
        x = (1-leak)*x + leak*f
        S[t]=x
    return S

def reservoir_eval(N, u, y, seed=1, K=20, rho=0.9, wash=200, alpha=1e-3, gain=0.5, act='tanh'):
    t0=time.time()
    uc = (u - u.mean()).astype(np.float32)                       # center the input (important)
    W,_ = build_reservoir(N, K, rho, seed)
    S = run_reservoir(W, uc, gain=gain, act=act)
    X = np.hstack([S, uc[:,None], np.ones((len(u),1),np.float32)])[wash:]   # states + raw input + bias
    yt = y[wash:]
    ntr = int(0.7*len(X))
    A = X[:ntr].T@X[:ntr] + alpha*np.eye(X.shape[1]); b = X[:ntr].T@yt[:ntr]
    w = np.linalg.solve(A, b)
    pred = X[ntr:]@w
    tag = "reservoir" + ("(Dale)" if act=='relu' else "")
    return dict(model=f"{tag} N={N}", params=N+2, nrmse=nrmse(pred, yt[ntr:]),
                secs=round(time.time()-t0,2))

# ------------------------------------------------- transformer baseline (PyTorch/GPU)
def transformer_eval(u, y, d_model=64, layers=2, heads=4, win=20, steps=400, seed=1):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # windowed regression: predict y[t] from u[t-win+1 .. t]
    Xs=[]; Ys=[]
    for t in range(win, len(u)):
        Xs.append(u[t-win:t]); Ys.append(y[t])
    X=torch.tensor(np.array(Xs)[:,:,None],dtype=torch.float32); Y=torch.tensor(np.array(Ys),dtype=torch.float32)
    ntr=int(0.7*len(X)); Xtr,Ytr,Xte,Yte=X[:ntr].to(dev),Y[:ntr].to(dev),X[ntr:].to(dev),Y[ntr:].to(dev)
    class TF(nn.Module):
        def __init__(s):
            super().__init__()
            s.inp=nn.Linear(1,d_model); s.pos=nn.Parameter(torch.randn(win,d_model)*0.02)
            el=nn.TransformerEncoderLayer(d_model,heads,d_model*2,batch_first=True,dropout=0.0)
            s.enc=nn.TransformerEncoder(el,layers); s.out=nn.Linear(d_model,1)
        def forward(s,x):
            h=s.inp(x)+s.pos; h=s.enc(h); return s.out(h[:,-1,:]).squeeze(-1)
    m=TF().to(dev); opt=torch.optim.Adam(m.parameters(),1e-3); lossf=nn.MSELoss()
    t0=time.time()
    for _ in range(steps):
        opt.zero_grad(); l=lossf(m(Xtr),Ytr); l.backward(); opt.step()
    m.eval()
    with torch.no_grad(): pred=m(Xte).cpu().numpy()
    P=sum(p.numel() for p in m.parameters())
    return dict(model=f"transformer d{d_model}x{layers}", params=P,
                nrmse=nrmse(pred, Yte.cpu().numpy()), secs=round(time.time()-t0,2), device=dev)

# ------------------------------------------------------------------------- main
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--reservoir_sizes", type=int, nargs="+", default=[100,300,1000,3000])
    ap.add_argument("--skip_transformer", action="store_true")
    a=ap.parse_args()
    u,y = narma10(a.T)
    rows=[]
    print(f"{'model':<22}{'params':>10}{'NRMSE':>9}{'secs':>8}")
    for N in a.reservoir_sizes:
        r=reservoir_eval(N,u,y); rows.append(r)
        print(f"{r['model']:<22}{r['params']:>10}{r['nrmse']:>9.4f}{r['secs']:>8}")
    if not a.skip_transformer:
        for dm,ly in [(32,2),(64,2),(128,3)]:
            try:
                r=transformer_eval(u,y,d_model=dm,layers=ly); rows.append(r)
                print(f"{r['model']:<22}{r['params']:>10}{r['nrmse']:>9.4f}{r['secs']:>8}  ({r['device']})")
            except Exception as e:
                print(f"transformer d{dm}x{ly}: FAILED ({e})")
    print("\nLower NRMSE = better. Compare error vs. params vs. secs — that's the Pareto trade-off.")

if __name__=="__main__":
    main()
