import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.special import kve as kv
plt.rcParams.update({'font.size': 9, 'savefig.dpi': 300})
F=np.array(json.load(open('../data/rel_forcing.json'))); R=json.load(open('../data/rel_compare.json'))
fig,ax=plt.subplots(1,3,figsize=(10.5,3.1))
m=F[:,0]<=10
ax[0].semilogx(F[m,0],F[m,1]/F[m,1].max(),'o-',ms=3,label=r'$F^2(T)/F^2_{max}$'); ax[0].semilogx(F[m,0],F[m,2]/F[m,2].max(),'s-',ms=3,label=r'$s(T)/s_{max}$')
ax[0].set_xlabel('$T/m$'); ax[0].legend(fontsize=8); ax[0].set_title('forcing, 3D Jüttner',fontsize=9)
for G,c in (('3','C0'),('30','C1'),('300','C2')):
    r=R[G]['rows']; a=[x['a'] for x in r]; ax[1].semilogy(a,[max(x['S'],1e-5) for x in r],'o-',ms=3,color=c,label=f'$n_0\\sigma={G}$ DSMC'); ax[1].semilogy(a,[x['Sqs'] for x in r],'--',color=c)
ax[1].set_xlabel('$a$'); ax[1].set_ylabel('$S$'); ax[1].legend(fontsize=7); ax[1].set_ylim(1e-4,0.3)
# frozen prediction for G0=3: ∫ s(T) dlog a
r=R['3']['rows']; a=np.array([x['a'] for x in r]); T=np.array([x['T'] for x in r]); s=np.interp(T,F[:,0],F[:,2]); Sfro=np.trapezoid(s,np.log(a)); print('frozen prediction',Sfro,'measured',r[-1]['S'])
for G,c in (('3','C0'),('300','C2')):
    h=np.array(R[G]['hist']); e=np.array(R[G]['edges']); pc=0.5*(e[1:]+e[:-1]); T=R[G]['Tfinal']
    ax[2].semilogy(pc,h,color=c,label=f'$n_0\\sigma={G}$, $T={T:.2f}$')
    w=pc**2*np.exp(-(np.sqrt(1+pc**2)-1)/T); w/=np.trapezoid(w,pc); ax[2].semilogy(pc,w,'--',color='k',lw=0.8)
ax[2].set_xlabel('$|p|$'); ax[2].set_ylabel('density'); ax[2].legend(fontsize=7); ax[2].set_ylim(1e-4,3)
plt.tight_layout(); plt.savefig('../figures/fig_rel.png'); print('ok')
