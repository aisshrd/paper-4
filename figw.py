import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})
d=json.load(open('../data/wimp_p5.json')); r=d['rows']; a=np.array([x['a'] for x in r])
# ODE for p=7 (mu0=400) analytically via RK4
def ode(p,mu0,t0=1.0,aend=14.0,dt=1e-3):
    t,Tc,Tb0=0.0,1.0,1.0; rows=[]
    H=lambda t:1/(2*(t0+t)); A=lambda t:(1+t/t0)**0.5
    while A(t)<aend:
        f=lambda tt,T:-2*H(tt)*T+0.5*mu0*A(tt)**(-p)*(Tb0/A(tt)-T)
        k1=f(t,Tc);k2=f(t+dt/2,Tc+dt/2*k1);k3=f(t+dt/2,Tc+dt/2*k2);k4=f(t+dt,Tc+dt*k3);Tc+=dt/6*(k1+2*k2+2*k3+k4);t+=dt
        rows.append((A(t),Tc,1-Tc*A(t),mu0*A(t)**(-p)/2/H(t)))
    return np.array(rows)
o5=ode(5,100); o7=ode(7,400)
fig,ax=plt.subplots(1,3,figsize=(10.5,3.1))
ax[0].loglog(o5[:,0],o5[:,1]*o5[:,0],'C0-',label=r'$p=5$: $T_\chi/T_b$ (ODE)'); ax[0].loglog(a,[x['Tchi']*x['a'] for x in r],'C0o',ms=3,label='$p=5$: кинетика')
ax[0].loglog(o7[:,0],o7[:,1]*o7[:,0],'C1-',label='$p=7$ (ODE)'); ax[0].loglog(o5[:,0],1/o5[:,0],'k:',label=r'$a^{-1}$ (полное decoupling)')
ax[0].axvline(d['a_kd'],color='C0',ls='--',lw=0.8); ax[0].axvline(400**(1/5),color='C1',ls='--',lw=0.8)
ax[0].set_xlabel('$a$'); ax[0].set_ylabel(r'$T_\chi/T_b$'); ax[0].legend(fontsize=7)
m=o5[:,3]>0.5
ax[1].loglog(o5[m,3],o5[m,2],'C0-',label=r'$\delta$ exact, $p=5$'); mm=o7[:,3]>0.5; ax[1].loglog(o7[mm,3],o7[mm,2],'C1-',label='$p=7$')
ax[1].loglog(o5[m,3],1/o5[m,3],'k--',label=r'$\delta_{qs}=H/(\mu/2)$'); ax[1].loglog([x['mu_over_H'] for x in r],[x['delta'] for x in r],'C0o',ms=3,label='кинетика')
ax[1].set_xlabel(r'$(\mu/2)/H$'); ax[1].set_ylabel(r'$\delta=1-T_\chi/T_b$'); ax[1].invert_xaxis(); ax[1].legend(fontsize=7)
ax[2].semilogx([x['mu_over_H'] for x in r],[x['a4'] for x in r],'o-',ms=3,label='$a_4$'); ax[2].semilogx([x['mu_over_H'] for x in r],[x['a6'] for x in r],'s-',ms=3,label='$a_6$')
ax[2].axvline(1,color='k',ls=':'); ax[2].invert_xaxis(); ax[2].set_xlabel(r'$(\mu/2)/H$'); ax[2].set_ylabel('коэффициенты формы'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig('../figures/fig_wimp.png')
json.dump(dict(o5=o5[::100].tolist(),o7=o7[::100].tolist()),open('../data/wimp_ode.json','w'))
for x in r[::3]: print(f"a={x['a']:5.2f} (μ/2)/H={x['mu_over_H']:7.2f} δ={x['delta']:+.4f} δqs={x['delta_qs']:.4f} Tchi/ODE={x['Tchi']/x['Tode']:.4f} a4={x['a4']:+.4f} a6={x['a6']:+.4f}")
print('p7 a_kd',400**(1/5), 'asymptotic Tchi*a p5', o5[-1,1]*o5[-1,0]**2, 'p7', o7[-1,1]*o7[-1,0]**2)
