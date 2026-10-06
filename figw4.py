import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'savefig.dpi': 300})
d=json.load(open('../data/wimp_p4_mu21.json')); r=d['rows']; O=json.load(open('../data/wimp_ode4.json')); o4=np.array(O['o4']); o6=np.array(O['o6'])
a=np.array([x['a'] for x in r]); akd=21.5**0.5
fig,ax=plt.subplots(1,3,figsize=(10.5,3.1))
ax[0].loglog(o4[:,0],o4[:,1]*o4[:,0],'C0-',label='$p=4$, exact ODE'); ax[0].loglog(a,[x['Tchi']*x['a'] for x in r],'C0o',ms=3,label='$p=4$, kinetic')
ax[0].loglog(o6[:,0],o6[:,1]*o6[:,0],'C1-',label='$p=6$, exact ODE'); ax[0].loglog(o4[:,0],1/o4[:,0],'k:',label=r'$a^{-1}$ (free streaming)')
ax[0].axvline(akd,color='gray',ls='--',lw=0.8); ax[0].set_xlabel('$a$'); ax[0].set_ylabel(r'$T_\chi/T_b$'); ax[0].legend(fontsize=7)
m4=(o4[:,3]>0.3)&(o4[:,0]>1.15)&(o4[:,3]<30); m6=(o6[:,3]>0.3)&(o6[:,0]>1.15)&(o6[:,3]<30)
ax[1].loglog(o4[m4,3],o4[m4,2],'C0-',label=r'$\delta$, $p=4$'); ax[1].loglog(o6[m6,3],o6[m6,2],'C1-',label=r'$\delta$, $p=6$')
xx=np.logspace(np.log10(0.3),np.log10(25),100); ax[1].loglog(xx,1/(xx+1),'k:',label=r'leading: $\mathcal{H}/(\gamma_2+\mathcal{H})$'); ax[1].loglog(xx,1/(xx+3),'k--',label=r'drift-corrected, $p=4$: $\mathcal{H}/(\gamma_2+3\mathcal{H})$')
rr=[x for x in r if x['a']>1.1]; ax[1].loglog([x['mu_over_H'] for x in rr],[x['delta'] for x in rr],'C0o',ms=3,label='kinetic, $p=4$')
ax[1].set_xlabel(r'$\gamma_2/\mathcal{H}$'); ax[1].set_ylabel(r'$\delta=1-T_\chi/T_b$'); ax[1].invert_xaxis(); ax[1].legend(fontsize=7)
rs=[x for x in r if x['a']<=6.5]; ax[2].semilogx([x['mu_over_H'] for x in rs],[x['a4'] for x in rs],'o-',ms=3,label='$a_4$'); ax[2].semilogx([x['mu_over_H'] for x in rs],[x['a6'] for x in rs],'s-',ms=3,label='$a_6$')
ax[2].axvline(1,color='k',ls=':'); ax[2].invert_xaxis(); ax[2].set_xlabel(r'$\gamma_2/\mathcal{H}$'); ax[2].set_ylabel('shape coefficients'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig('../figures/fig_wimp.png')
Wd=json.load(open('../data/snap_wimp_p4.json')); W=np.array(Wd['W']); cm=plt.cm.viridis
fig,ax=plt.subplots(1,2,figsize=(10,3.4))
for i,s in enumerate(Wd['snaps']):
    g=np.array(s['g']); c=cm(i/(len(Wd['snaps'])-1)); ax[0].plot(W,g,color=c,label=f"$a={s['a']:.1f}$, $T_\\chi/T_b={s['Tchi']/s['Tb']:.2f}$")
    Mb=np.exp(-W**2/(2*s['Tb']))/np.sqrt(2*np.pi*s['Tb']); ax[0].plot(W,Mb,'--',color=c,lw=0.7)
ax[0].set_xlim(-2.5,2.5); ax[0].set_xlabel('$w$'); ax[0].set_ylabel('$g(w)$'); ax[0].legend(fontsize=6.5)
for i,s in enumerate(Wd['snaps']):
    g=np.array(s['g']); T=s['Tchi']; c=cm(i/(len(Wd['snaps'])-1)); ax[1].semilogy(W/np.sqrt(T),g*np.sqrt(T),color=c,label=f"$a={s['a']:.1f}$")
ax[1].semilogy(W,np.exp(-W**2/2)/np.sqrt(2*np.pi),'k--',lw=1,label='Gaussian'); ax[1].set_xlim(-4.5,4.5); ax[1].set_ylim(1e-5,1); ax[1].set_xlabel(r'$w/\sqrt{T_\chi}$'); ax[1].legend(fontsize=6.5)
plt.tight_layout(); plt.savefig('../figures/fig_prof_wimp.png'); print('figs ok')
