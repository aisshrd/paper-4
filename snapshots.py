import numpy as np, json, sys
which = sys.argv[1]
if which == 'wimp':
    src = open('wimp.py').read().split('p = float')[0]
    src = src.replace("'L, H = 8.0, 0.08'", "'L, H = 8.0, 0.1'").replace("M_TH = 48", "M_TH = 32")
    exec(src)
    p, mu0, G0, t0, Tb0, aend = 5.0, 100.0, 5.0, 1.0, 1.0, 12.0
    Hub = lambda t: 1/(2*(t0+t)); a_of = lambda t: (1+t/t0)**0.5; mu_of = lambda t: mu0*a_of(t)**(-p); Tb_of = lambda t: Tb0/a_of(t)
    g = gauss(Tb0); t = 0.0; snaps = []; targets = [1.0, 2.0, 3.0, 4.64, 6.0, 8.0, 11.0]; k = 0
    while a_of(t) < aend:
        Hh, mu, G = Hub(t), mu_of(t), G0*a_of(t)**-3
        dt = min(0.1/max(mu, 1e-9), 0.1/max(G, 1e-9), 0.3*h/(Hh*L), 0.05*(t0+t))
        def rhs(gg, tt):
            Hh_, mu_, G_, Tb_ = Hub(tt), mu_of(tt), G0*a_of(tt)**-3, Tb_of(tt)
            flux = -Hh_*W*gg; tr = np.zeros_like(gg); tr[1:-1] = -(flux[2:]-flux[:-2])/(2*h)
            return tr + G_*Q_coll(gg) + mu_*thermostat(gg, Tb_)
        k1 = rhs(g, t); k2 = rhs(g+dt/2*k1, t+dt/2); k3 = rhs(g+dt/2*k2, t+dt/2); k4 = rhs(g+dt*k3, t+dt)
        g = np.maximum(g + dt/6*(k1+2*k2+2*k3+k4), 0); g /= h*g.sum(); t += dt
        if k < len(targets) and a_of(t) >= targets[k]:
            Tc = h*np.sum(W**2*g); snaps.append(dict(a=float(a_of(t)), Tb=float(Tb_of(t)), Tchi=float(Tc), g=g.tolist())); k += 1
            print(f'snapshot a={a_of(t):.2f} Tb={Tb_of(t):.3f} Tchi={Tc:.3f}', flush=True)
    json.dump(dict(W=W.tolist(), snaps=snaps, a_kd=(mu0*t0)**(1/(p-2))), open('../data/snap_wimp.json', 'w'))
if which == 'flrw':
    src = open('flrw.py').read().split('# ---------- 1. функция форсинга')[0]
    exec(src)
    G0, t0, T0, tend = 4.0, 1.0, 4.0, 150.0
    g = gauss(T0); t = 0.0; snaps = []; targets = [1.0, 1.5, 2.2, 3.2, 4.7, 7.0, 10.0]; k = 0
    Hub = lambda t: 1/(2*(t0+t)); Gam = lambda t: G0*(1+t/t0)**(-1.5)
    while t < tend:
        Hh, G = Hub(t), Gam(t); dt = min(0.25, 0.4*h/(Hh*L), 0.2/max(G, 1e-9), 0.05*(t0+t))
        def rhs(gg, tt): return transport(gg, Hub(tt)) + Gam(tt)*Q_coll(gg)
        k1 = rhs(g, t); k2 = rhs(g+dt/2*k1, t+dt/2); k3 = rhs(g+dt/2*k2, t+dt/2); k4 = rhs(g+dt*k3, t+dt)
        g = np.maximum(g + dt/6*(k1+2*k2+2*k3+k4), 0); g /= h*g.sum(); t += dt
        a = (1+t/t0)**0.5
        if k < len(targets) and a >= targets[k]:
            T = h*np.sum(W**2*g); snaps.append(dict(a=float(a), T=float(T), ratio=float(G*0.25/Hh), g=g.tolist())); k += 1
            print(f'snapshot a={a:.2f} T={T:.3f} Γγ1/H={G*0.25/Hh:.2f}', flush=True)
    json.dump(dict(W=W.tolist(), snaps=snaps), open('../data/snap_flrw.json', 'w'))
