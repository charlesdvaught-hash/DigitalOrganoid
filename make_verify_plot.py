import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

data = json.load(open('sweep_results.json')) + json.load(open('results_2e5.json'))
agg = defaultdict(lambda: defaultdict(list))
for r in data:
    if r.get('status')!='success' or abs(r['rho']-0.95)>1e-6: continue
    agg[(r['N'],r['K'])]['mc'].append(r['total_mc'])

for K,color in [(50,'#2f7a4d'),(100,'#c9803a')]:
    Ns=sorted(n for (n,k) in agg if k==K)
    mc=[np.mean(agg[(n,K)]['mc']) for n in Ns]
    sd=[np.std(agg[(n,K)]['mc']) for n in Ns]
    plt.errorbar(Ns,mc,yerr=sd,marker='o',color=color,label=f'K={K}',capsize=3)
# log fit on K=50
Ns=np.array(sorted(n for (n,k) in agg if k==50))
mc=np.array([np.mean(agg[(n,50)]['mc']) for n in Ns])
b,a=np.polyfit(np.log10(Ns),mc,1)
xf=np.logspace(2,5.4,50)
plt.plot(xf,a+b*np.log10(xf),'--',color='#6ea8ff',alpha=.8,label=f'log fit K=50: MC={a:.2f}+{b:.2f}·log₁₀N')
plt.xscale('log'); plt.xlabel('Network size N (log scale)'); plt.ylabel('Memory Capacity (MC)')
plt.title('Memory Capacity scales logarithmically with N (ρ=0.95)')
plt.grid(True,alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig('plots/mc_logfit.png',dpi=110,facecolor='white')
print('log fit K=50: MC =',round(a,3),'+',round(b,3),'* log10(N)')
print('R2 predicted MC at N=100:',round(a+b*2,2),' at N=1e5:',round(a+b*5,2))
