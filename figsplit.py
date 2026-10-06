import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'savefig.dpi': 300})
Wd=json.load(open('../data/snap_wimp.json')); Fd=json.load(open('../data/snap_flrw.json')); W=np.array(Wd['W']); WF=np.array(Fd['W']); cm=plt.cm.viridis
fig,ax=plt.subplots(1,2,figsize=(10,3.4))
for i,s in enumerate(Wd['snaps']):
    g=np.array(s['g']); c=cm(i/(len(Wd['snaps'])-1)); ax[0].plot(W,g,color=c,label=f"$a={s['a']:.1f}$, $T_\\chi/T_b={s['Tchi']/s['Tb']:.2f}$")
    Mb=np.exp(-W**2/(2*s['Tb']))/np.sqrt(2*np.pi*s['Tb']); ax[0].plot(W,Mb,'--',color=c,lw=0.7)
ax[0].set_xlim(-2.5,2.5); ax[0].set_xlabel('$w$'); ax[0].set_ylabel('$g(w)$'); ax[0].legend(fontsize=6.5)
for i,s in enumerate(Wd['snaps']):
    g=np.array(s['g']); T=s['Tchi']; c=cm(i/(len(Wd['snaps'])-1)); ax[1].semilogy(W/np.sqrt(T),g*np.sqrt(T),color=c,label=f"$a={s['a']:.1f}$")
ax[1].semilogy(W,np.exp(-W**2/2)/np.sqrt(2*np.pi),'k--',lw=1,label='Gaussian'); ax[1].set_xlim(-4.5,4.5); ax[1].set_ylim(1e-5,1); ax[1].set_xlabel(r'$w/\sqrt{T_\chi}$'); ax[1].legend(fontsize=6.5)
plt.tight_layout(); plt.savefig('../figures/fig_prof_wimp.png')
fig,ax=plt.subplots(1,2,figsize=(10,3.4))
for i,s in enumerate(Fd['snaps']):
    g=np.array(s['g']); c=cm(i/(len(Fd['snaps'])-1)); ax[0].plot(WF,g,color=c,label=f"$a={s['a']:.1f}$, $T={s['T']:.2f}$, $\\Gamma\\gamma_1/\\mathcal{{H}}={s['ratio']:.1f}$")
ax[0].set_xlim(-5,5); ax[0].set_xlabel('$w$'); ax[0].set_ylabel('$g(w)$'); ax[0].legend(fontsize=6.5)
for i,s in enumerate(Fd['snaps']):
    g=np.array(s['g']); T=s['T']; c=cm(i/(len(Fd['snaps'])-1)); ax[1].semilogy(WF/np.sqrt(T),g*np.sqrt(T),color=c,label=f"$a={s['a']:.1f}$")
ax[1].semilogy(WF,np.exp(-WF**2/2)/np.sqrt(2*np.pi),'k--',lw=1,label='Gaussian'); ax[1].set_xlim(-4.5,4.5); ax[1].set_ylim(1e-5,1); ax[1].set_xlabel(r'$w/\sqrt{T}$'); ax[1].legend(fontsize=6.5)
plt.tight_layout(); plt.savefig('../figures/fig_prof_flrw.png'); print('ok')
