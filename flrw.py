"""
Модель Каца в расширяющейся Вселенной (ФЛРУ, фон задан, обратного влияния нет).
Переменная w: кинетическая энергия E-1 = w^2/2 (m=1), знак = знак импульса. Столкновения Каца --- повороты в w
(сохраняют E1+E2), равновесие --- гауссиан в w с дисперсией T (юттнеровский вес e^{-E/T}).
Транспорт (красное смещение p -> p/a): dw/dt = c(w) = -H(t) w (E+1)/(2E), E = 1 + w^2/2.
Нормированное g (∫g dw = 1):   ∂_t g + ∂_w(c g) = Γ(t) Q_Kac[g],  Γ(t) = λ n(t), n ∝ a^{-3}.
"""
import numpy as np, json, time
from numpy.polynomial.hermite_e import hermeval
from math import factorial
kg = open('kac_geom.py').read().split('results = {}')[0].replace('L, H = 7.0, 0.07', 'L, H = 10.0, 0.1')
exec(kg)
W = V; h = H                      # сетка по w
out = {}

def Efun(w): return 1 + w**2/2
def cvel(w, Hub): return -Hub*w*(Efun(w) + 1)/(2*Efun(w))

def transport(g, Hub):
    """-∂_w(c g), центральные разности, нулевой поток на краях"""
    flux = cvel(W, Hub)*g
    d = np.zeros_like(g); d[1:-1] = -(flux[2:] - flux[:-2])/(2*h)
    return d

def gauss(T): return np.exp(-W**2/(2*T))/np.sqrt(2*np.pi*T)

def hermite_w(n, T):
    """нормированные полиномы Эрмита в масштабе T: ортонормированы относительно gauss(T)"""
    c = np.zeros(n+1); c[n] = 1; return hermeval(W/np.sqrt(T), c)/np.sqrt(factorial(n))

def project_perp(v, g):
    """вычесть компоненты, меняющие массу и энергию: v_perp = v - (α + β w^2) g с ∫v_perp = ∫w^2 v_perp = 0"""
    m0, m2, m4 = h*g.sum(), h*np.sum(W**2*g), h*np.sum(W**4*g)
    b0, b2 = h*v.sum(), h*np.sum(W**2*v)
    A = np.array([[m0, m2], [m2, m4]]); al, be = np.linalg.solve(A, [b0, b2])
    return v - (al + be*W**2)*g

def forcing_modes(T, nmax=10):
    """коэффициенты Эрмита (в масштабе T) нормальной части транспорта гауссиана, на единицу H"""
    g = gauss(T); v = project_perp(transport(g, 1.0), g)
    return [h*np.sum(v*hermite_w(n, T)) for n in range(nmax+1)]

# ---------- 1. функция форсинга F(T): норма нормальной части в метрике Фишера (на единицу H) ----------
Ts = np.concatenate([np.linspace(0.05, 1, 20), np.linspace(1.2, 10, 23)])
Fmodes = {float(T): forcing_modes(T) for T in Ts}
F2 = {T: sum(m[n]**2 for n in range(3, 11)) for T, m in Fmodes.items()}   # моды n>=3 (1,2 исключены проекцией; n=0 ~0)
out['forcing'] = dict(T=list(F2.keys()), F2=list(F2.values()), modes=[Fmodes[T][:9] for T in Fmodes])
Tmax = max(F2, key=F2.get); print('forcing F^2(T) peak at T =', Tmax, ' F^2 =', F2[Tmax], flush=True)
for T in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
    k = min(F2, key=lambda x: abs(x-T)); print(f'T={k:.2f}: F^2={F2[k]:.5f}  modes4,6,8={np.round(Fmodes[k][4:9:2],4)}')

# ---------- 2. эволюция: a=(1+t/t0)^{1/2}, H=1/(2(t0+t)), Γ=Γ0 a^{-3} ----------
def run(G0, t0, T0, tend, gamma1=0.25):
    g = gauss(T0); t = 0.0; rows = []; tout = 0.0
    def Hub(t): return 1/(2*(t0 + t))
    def Gam(t): return G0*(1 + t/t0)**(-1.5)
    while t < tend:
        Hh, G = Hub(t), Gam(t)
        dt = min(0.25, 0.4*h/(Hh*L), 0.2/max(G, 1e-9), 0.05*(t0 + t))
        def rhs(gg, tt): return transport(gg, Hub(tt)) + Gam(tt)*Q_coll(gg)
        k1 = rhs(g, t); k2 = rhs(g + dt/2*k1, t + dt/2); k3 = rhs(g + dt/2*k2, t + dt/2); k4 = rhs(g + dt*k3, t + dt)
        g = np.maximum(g + dt/6*(k1 + 2*k2 + 2*k3 + k4), 0); g /= h*g.sum(); t += dt
        if t >= tout:
            T = h*np.sum(W**2*g); G_T = gauss(T); m = (g > 1e-30) & (G_T > 1e-30)
            Erel = h*np.sum(g[m]*np.log(g[m]/G_T[m]))
            # квазистационарное предсказание: delta_n = H F_n(T)/(Γ γ_n), E_rel ≈ 1/2 Σ delta_n^2
            fm = forcing_modes(T); gam = [1 - 2*np.mean(np.cos(TH)**n) for n in range(11)]
            qs = 0.5*sum((Hub(t)*fm[n]/(G*gam[n]))**2 for n in range(3, 11))
            # фактические коэффициенты Эрмита
            a_n = [h*np.sum(g*hermite_w(n, T)) for n in range(11)]
            Dcoll = G*np.sum(np.where(Wgeo(g) > 0, Wgeo(g)*rfield(g)*np.sinh(np.where(Wgeo(g) > 0, rfield(g), 0)), 0))*WTH*h*h
            rows.append(dict(t=float(t), a=float((1 + t/t0)**0.5), T=float(T), Hub=float(Hub(t)), Gam=float(G), Erel=float(Erel), Eqs=float(qs),
                             a4=float(a_n[4]), a6=float(a_n[6]), ratio=float(G*gamma1/Hub(t)), D=float(Dcoll)))
            print(f"t={t:8.2f} a={rows[-1]['a']:6.2f} T={T:.4f} Γγ1/H={rows[-1]['ratio']:8.3f} Erel={Erel:.3e} Eqs={qs:.3e} a4={a_n[4]:+.4f} a6={a_n[6]:+.4f}", flush=True)
            tout = t*1.3 + 0.5
    return rows, g

t0 = time.time()
rowsA, gA = run(G0=40.0, t0=1.0, T0=4.0, tend=150.0)
print('run A done', time.time()-t0, flush=True)
np.save('../data/flrw_gA.npy', gA)
rowsB, gB = run(G0=400.0, t0=1.0, T0=4.0, tend=150.0)
rowsC, gC = run(G0=4.0, t0=1.0, T0=4.0, tend=150.0)
np.save('../data/flrw_gC.npy', gC)
print('run B done', time.time()-t0, flush=True)
np.save('../data/flrw_gB.npy', gB)
out['runA'] = rowsA; out['runB'] = rowsB; out['runC'] = rowsC
json.dump(out, open('../data/flrw.json', 'w'), indent=1)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
ax[0].plot(list(F2.keys()), list(F2.values()), 'o-', ms=3); ax[0].set_xscale('log'); ax[0].set_xlabel('$T/m$'); ax[0].set_ylabel('$F^2(T)$'); ax[0].set_title('форсинг транспорта', fontsize=9)
for rows, lab in ((rowsC, r'$\Gamma_0=4$'), (rowsA, r'$\Gamma_0=40$'), (rowsB, r'$\Gamma_0=400$')):
    r = rows; ax[1].loglog([x['a'] for x in r], [x['Erel'] for x in r], 'o-', ms=3, label=lab+' (симуляция)')
    ax[1].loglog([x['a'] for x in r], [x['Eqs'] for x in r], '--', label=lab+' (квазистац.)')
ax[1].set_xlabel('$a$'); ax[1].set_ylabel(r'$E_{rel}$'); ax[1].legend(fontsize=7); ax[1].set_ylim(1e-7, 1)
for gg, lab in ((gC, r'$\Gamma_0=4$'), (gA, r'$\Gamma_0=40$'), (gB, r'$\Gamma_0=400$')):
    Tg = h*np.sum(W**2*gg); ax[2].semilogy(W/np.sqrt(Tg), gg*np.sqrt(Tg), label=lab)
ax[2].semilogy(W, np.exp(-W**2/2)/np.sqrt(2*np.pi), 'k--', label='гауссиан')
ax[2].set_xlim(-4, 4); ax[2].set_ylim(1e-5, 1); ax[2].set_xlabel(r'$w/\sqrt{T}$'); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig('../figures/fig_flrw.png'); print('done')
