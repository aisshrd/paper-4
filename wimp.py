"""
Кинетическое decoupling тяжёлой частицы (NR, m >> T) от релятивистской ванны, T_b = T_b0/a.
Столкновения с ванной: термостат Каца (часть II), интенсивность μ(T_b) = μ0 (T_b/T_b0)^p, p=5 (контактное взаимодействие).
Самостолкновения: Γ(t)=Γ0 a^{-3} (кацевские). Транспорт NR: dw/dt = -H w, H=1/(2(t0+t)), a=(1+t/t0)^{1/2}.
   ∂_t g + ∂_w(-H w g) = Γ Q_Kac[g] + μ T_b[g],   T_b[g](u) = ∫dθ/2π ∫dw [g(u c - w s) M_b(u s + w c) - g(u) M_b(w)]
Точная ODE для T_chi (термостатная мода, γ2 = μ/2):  dT_chi/dt = -2H T_chi + (μ/2)(T_b - T_chi).
"""
import numpy as np, json, sys
from numpy.polynomial.hermite_e import hermegauss, hermeval
from math import factorial
kg = open('kac_geom.py').read().split('results = {}')[0].replace('L, H = 7.0, 0.07', 'L, H = 8.0, 0.08')
exec(kg)
W = V; h = H
xg, wg = hermegauss(24); wg = wg/np.sqrt(2*np.pi)
def thermostat(g, Tb):
    """μ-часть без множителя μ: ∫dθ/2π ∫dw [g(u c - w s) M_b(u s + w c) - g(u) M_b(w)], квадратура по w на сетке"""
    out = np.zeros_like(g)
    for t in TH:
        c, sn = np.cos(t), np.sin(t)
        up = W[:, None]*c - W[None, :]*sn; wp = W[:, None]*sn + W[None, :]*c
        Mb = np.exp(-wp**2/(2*Tb))/np.sqrt(2*np.pi*Tb)
        out += h*np.sum(interp(g, up)*Mb, axis=1)
    return out/M_TH - g
def gauss(T): return np.exp(-W**2/(2*T))/np.sqrt(2*np.pi*T)
def hermT(n, T):
    c = np.zeros(n+1); c[n] = 1; return hermeval(W/np.sqrt(T), c)/np.sqrt(factorial(n))

p = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
mu0 = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
G0 = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
t0, Tb0, aend = 1.0, 1.0, 14.0
def Hub(t): return 1/(2*(t0 + t))
def a_of(t): return (1 + t/t0)**0.5
def mu_of(t): return mu0*a_of(t)**(-p)
def Tb_of(t): return Tb0/a_of(t)

# ---- точная ODE для температуры (RK4) и квазистационарное δ ----
def ode_T(dt=1e-3):
    t, Tc = 0.0, Tb0; rows = []
    while a_of(t) < aend:
        f = lambda tt, T: -2*Hub(tt)*T + 0.5*mu_of(tt)*(Tb_of(tt) - T)
        k1 = f(t, Tc); k2 = f(t+dt/2, Tc+dt/2*k1); k3 = f(t+dt/2, Tc+dt/2*k2); k4 = f(t+dt, Tc+dt*k3)
        Tc += dt/6*(k1+2*k2+2*k3+k4); t += dt
        rows.append((t, a_of(t), Tc))
    return np.array(rows)
ode = ode_T()
# a_kd: μ/2 = H  <=>  mu0 a^{-p}/2 = 1/(2 t0 a^2)  <=> a^{p-2} = mu0 t0
a_kd = (mu0*t0)**(1/(p-2)); print(f'p={p} mu0={mu0}: a_kd = {a_kd:.3f}, T_kd/T_b0 = {1/a_kd:.4f}', flush=True)

# ---- полная кинетика ----
g = gauss(Tb0); t = 0.0; rows = []; tout = 0.0
while a_of(t) < aend:
    Hh, mu, G, Tb = Hub(t), mu_of(t), G0*a_of(t)**-3, Tb_of(t)
    dt = min(0.1/max(mu, 1e-9), 0.1/max(G, 1e-9), 0.3*h/(Hh*L), 0.05*(t0+t))
    def rhs(gg, tt):
        Hh_, mu_, G_, Tb_ = Hub(tt), mu_of(tt), G0*a_of(tt)**-3, Tb_of(tt)
        flux = -Hh_*W*gg; tr = np.zeros_like(gg); tr[1:-1] = -(flux[2:]-flux[:-2])/(2*h)
        return tr + G_*Q_coll(gg) + mu_*thermostat(gg, Tb_)
    k1 = rhs(g, t); k2 = rhs(g+dt/2*k1, t+dt/2); k3 = rhs(g+dt/2*k2, t+dt/2); k4 = rhs(g+dt*k3, t+dt)
    g = np.maximum(g + dt/6*(k1+2*k2+2*k3+k4), 0); g /= h*g.sum(); t += dt
    if t >= tout:
        Tc = h*np.sum(W**2*g); a = a_of(t); Tb = Tb_of(t)
        Tode = np.interp(t, ode[:, 0], ode[:, 2])
        dqs = 2*Hub(t)/mu_of(t)
        a4 = h*np.sum(g*hermT(4, Tc)); a6 = h*np.sum(g*hermT(6, Tc))
        m = g > 1e-30; Erel = h*np.sum(g[m]*np.log(g[m]/gauss(Tc)[m]))
        rows.append(dict(t=float(t), a=float(a), Tb=float(Tb), Tchi=float(Tc), Tode=float(Tode), delta=float(1-Tc/Tb), delta_qs=float(dqs),
                         mu_over_H=float(mu_of(t)/2/Hub(t)), a4=float(a4), a6=float(a6), Erel=float(Erel)))
        print(f"a={a:6.2f} Tb={Tb:.4f} Tchi={Tc:.4f} (ODE {Tode:.4f}) δ={1-Tc/Tb:+.4f} δ_qs={dqs:.4f} (μ/2)/H={mu_of(t)/2/Hub(t):8.3f} a4={a4:+.4f} Erel={Erel:.2e}", flush=True)
        tout = t*1.2 + 0.3
json.dump(dict(p=p, mu0=mu0, G0=G0, a_kd=a_kd, rows=rows, ode=ode[::50].tolist(), g_final=g.tolist(), W=W.tolist()), open(f'../data/wimp_p{int(p)}.json', 'w'), indent=1)
