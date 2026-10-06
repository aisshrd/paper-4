import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'savefig.dpi': 300})
d=json.load(open('../data/flrw.json')); T=np.array(d['forcing']['T']); F2=np.array(d['forcing']['F2'])
W=np.arange(-10.0,10.0+1e-9,0.1)
fig,ax=plt.subplots(1,3,figsize=(10.5,3.1))
m=T>0.07; ax[0].plot(T[m],F2[m],'o-',ms=3); ax[0].set_xscale('log'); ax[0].set_xlabel('$T/m$'); ax[0].set_ylabel('$F^2(T)$'); ax[0].set_title('transport forcing',fontsize=9)
for key,lab in (('runC',r'$\Gamma_0=4$'),('runA',r'$\Gamma_0=40$'),('runB',r'$\Gamma_0=400$')):
    r=d[key]; ax[1].loglog([x['a'] for x in r],[x['Erel'] for x in r],'o-',ms=3,label=lab+' (simulation)'); ax[1].loglog([x['a'] for x in r],[x['Eqs'] for x in r],'--',label=lab+' (quasi-stationary)')
ax[1].set_xlabel('$a$'); ax[1].set_ylabel(r'$E_{\rm rel}$'); ax[1].legend(fontsize=6.5); ax[1].set_ylim(1e-7,1)
for f,lab in (('flrw_gC.npy',r'$\Gamma_0=4$'),('flrw_gA.npy',r'$\Gamma_0=40$'),('flrw_gB.npy',r'$\Gamma_0=400$')):
    g=np.load('../data/'+f); Tg=0.1*np.sum(W**2*g); ax[2].semilogy(W/np.sqrt(Tg),g*np.sqrt(Tg),label=lab)
ax[2].semilogy(W,np.exp(-W**2/2)/np.sqrt(2*np.pi),'k--',label='Gaussian'); ax[2].set_xlim(-4,4); ax[2].set_ylim(1e-5,1); ax[2].set_xlabel(r'$w/\sqrt{T}$'); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig('../figures/fig_flrw.png'); print('flrw ok')
