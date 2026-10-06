"""
Релятивистский газ твёрдых сфер (m=c=1), сечение σ=const, изотропное рассеяние в СЦМ, DSMC (NTC).
ФЛРУ: p -> p/a, n -> n/a^3, a=(1+t/t0)^{1/2}.  Γ := n σ (частота столкновений ~ Γ<v_M>).
Наблюдаемая формы: S = <E^2>/<E>^2 - (<E^2>/<E>^2)_J(T(<E>))  (отклонение от Юттнера при той же средней энергии).
"""
import numpy as np, json, sys, time
from scipy.special import kve as kv
from scipy.integrate import quad
from scipy.optimize import brentq
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)

# ---------- Юттнер в 3D ----------
def E_J(T):      # <E> для f ∝ e^{-E/T} (m=1)
    return 3*T + kv(1, 1/T)/kv(2, 1/T)
def E2_J(T):
    Z = lambda n: quad(lambda p: p**2*np.sqrt(1+p**2)**n*np.exp(-(np.sqrt(1+p**2)-1)/T), 0, 40*np.sqrt(T)+40, limit=200)[0]
    return Z(2)/Z(0)
def T_of_E(Emean):
    return brentq(lambda T: E_J(T) - Emean, 1e-3, 1e3)
def shape_J(T):  # (<E^2>/<E>^2)_J
    return E2_J(T)/E_J(T)**2

# ---------- форсинг: действие транспорта v = H f_J [3 - p^2/(E T)] на Юттнера ----------
def forcing(T):
    """F^2 = ||v_perp/f_J||^2_{f_J}/H^2 и источник s(T) = (d/dt S)|_transport / H"""
    pmax = 40*np.sqrt(T) + 40
    w = lambda p: p**2*np.exp(-(np.sqrt(1+p**2)-1)/T)
    Z = quad(w, 0, pmax, limit=200)[0]
    mom = lambda g: quad(lambda p: w(p)*g(p), 0, pmax, limit=200)[0]/Z
    E = lambda p: np.sqrt(1+p**2); v = lambda p: 3 - p**2/(E(p)*T)
    # v_perp = f_J [v - α - β E], ∫v_perp = ∫E v_perp = 0
    m1, m2 = mom(E), mom(lambda p: E(p)**2); v0, v1 = mom(v), mom(lambda p: v(p)*E(p))
    A = np.array([[1, m1], [m1, m2]]); al, be = np.linalg.solve(A, [v0, v1])
    F2 = mom(lambda p: (v(p) - al - be*E(p))**2)
    # источник для S: d<E^2>/dt|_tr = ∫ E^2 v f_J, d<E>/dt|_tr = ∫ E v f_J ; S = <E^2>/<E>^2 - shape_J(T(<E>))
    dE2 = mom(lambda p: E(p)**2*v(p)); dE1 = v1
    dT = dE1/((E_J(T*1.001) - E_J(T*0.999))/(0.002*T))
    dS = dE2/m1**2 - 2*m2*dE1/m1**3 - (shape_J(T*1.001) - shape_J(T*0.999))/(0.002*T)*dT
    return F2, dS

# ---------- DSMC ----------
def energies(P): return np.sqrt(1 + np.sum(P**2, axis=1))
def sample_juttner(N, T):
    out = []
    pmax = 40*np.sqrt(T) + 40; fmax = 1.0
    # отбор по p^2 e^{-(E-1)/T} / max
    pg = np.linspace(0, pmax, 4000); wg = pg**2*np.exp(-(np.sqrt(1+pg**2)-1)/T); fmax = wg.max()
    while len(out) < N:
        p = rng.random(2*N)*pmax; u = rng.random(2*N)
        keep = p[u*fmax < p**2*np.exp(-(np.sqrt(1+p**2)-1)/T)]; out.extend(keep.tolist())
    pm = np.array(out[:N]); n = rng.normal(size=(N, 3)); n /= np.linalg.norm(n, axis=1)[:, None]
    return pm[:, None]*n

def collide(P, nsigma, dt):
    """NTC с непересекающимися парами из случайной перестановки: кандидатов N/2 * (nσ) * v_max * dt, v_max=2"""
    N = len(P); ncand = min(rng.poisson(0.5*N*nsigma*2.0*dt), N//2)
    if ncand == 0: return 0
    perm = rng.permutation(N)[:2*ncand]; i, j = perm[:ncand], perm[ncand:]
    p1, p2 = P[i], P[j]; E1, E2 = energies(p1), energies(p2)
    dot4 = E1*E2 - np.sum(p1*p2, axis=1); vM = np.sqrt(np.maximum(dot4**2 - 1, 0))/(E1*E2)
    acc = rng.random(ncand) < vM/2.0
    i, j, p1, p2, E1, E2 = i[acc], j[acc], p1[acc], p2[acc], E1[acc], E2[acc]
    Ptot = p1 + p2; Etot = E1 + E2; M = np.sqrt(Etot**2 - np.sum(Ptot**2, axis=1)); ps = np.sqrt(np.maximum(M**2/4 - 1, 0))
    n = rng.normal(size=(len(i), 3)); n /= np.linalg.norm(n, axis=1)[:, None]
    pcm = ps[:, None]*n; Ecm = M/2
    beta = Ptot/Etot[:, None]; b2 = np.sum(beta**2, axis=1); gam = 1/np.sqrt(1 - b2)
    bp = np.sum(beta*pcm, axis=1)
    coef = np.where(b2 > 1e-14, (gam - 1)*bp/np.maximum(b2, 1e-14), 0.0) + gam*Ecm
    p1n = pcm + coef[:, None]*beta; p2n = Ptot - p1n
    P[i] = p1n; P[j] = p2n
    return len(i)

def shape_obs(P):
    E = energies(P); Em, E2m = E.mean(), (E**2).mean(); T = T_of_E(Em)
    return Em, T, E2m/Em**2 - shape_J(T)

# ---------- режимы ----------
mode = sys.argv[1]
out = {}
if mode == 'forcing':
    Ts = np.concatenate([np.linspace(0.05, 1, 12), np.linspace(1.5, 20, 15)]); res = []
    for T in Ts:
        F2, dS = forcing(T); res.append((float(T), float(F2), float(dS))); print(f'T={T:.3f} F2={F2:.5f} dS/H={dS:.5f}', flush=True)
    json.dump(res, open('../data/rel_forcing.json', 'w'), indent=1)

if mode == 'static':
    # скорость сжатия: релаксация S в статическом ящике; начальное: Юттнер с деформацией вдоль транспорта (шаг расширения без столкновений)
    N = int(sys.argv[3]); res = []
    for T in [float(x) for x in sys.argv[4].split(',')]:
        P = sample_juttner(N, T)
        # сильная деформация: применить "расширение" a: 1 -> 1.5 без столкновений (p -> p/1.5), это задаёт неравновесную форму
        P = P/1.5
        nsigma = 1.0; dt = 0.02; t = 0.0; S0 = shape_obs(P)[2]; traj = []
        while t < 8.0:
            collide(P, nsigma, dt); t += dt
            if abs(t/0.25 - round(t/0.25)) < 1e-9:
                Em, Tt, S = shape_obs(P); traj.append((t, S)); print(f'T0={T} t={t:.2f} S={S:+.5f}', flush=True)
        tr = np.array(traj); m = (tr[:, 0] >= 0.5) & (tr[:, 0] <= 3.0) & (np.abs(tr[:, 1]) > 1e-4)
        slope = np.polyfit(tr[m, 0], np.log(np.abs(tr[m, 1])), 1)[0] if m.sum() > 3 else np.nan
        Em, Tt, S = shape_obs(P)
        res.append(dict(T0=T, T_after=float(Tt), rate=float(-slope), traj=tr.tolist()))
        print(f'T0={T}: T_after={Tt:.3f}  gamma_S = {-slope:.4f} (per nσ)', flush=True)
    json.dump(res, open('../data/rel_static_%s.json' % sys.argv[2], 'w'), indent=1)

if mode == 'expand':
    N = int(sys.argv[3]); G0 = float(sys.argv[4]); t0 = 1.0; T0 = 4.0; tend = float(sys.argv[5])
    P = sample_juttner(N, T0); t = 0.0; a = 1.0; rows = []; tout = 0.0
    while t < tend:
        Hub = 1/(2*(t0 + t)); nsigma = G0*a**-3
        dt = min(0.02/Hub, 0.1/max(nsigma, 1e-9), 0.5)
        a_new = (1 + (t + dt)/t0)**0.5; P *= a/a_new; a = a_new
        collide(P, nsigma, dt); t += dt
        if t >= tout:
            Em, T, S = shape_obs(P); rows.append(dict(t=float(t), a=float(a), T=float(T), Hub=float(Hub), Gam=float(nsigma), S=float(S)))
            print(f'G0={G0} t={t:8.2f} a={a:6.2f} T={T:.4f} nσ/H={nsigma/Hub:8.3f} S={S:+.5f}', flush=True)
            tout = t*1.25 + 0.5
    hist, edges = np.histogram(np.linalg.norm(P, axis=1), bins=60, range=(0, 6*np.sqrt(T)+1), density=True)
    json.dump(dict(rows=rows, hist=hist.tolist(), edges=edges.tolist(), Tfinal=float(T)), open('../data/rel_expand_%s.json' % sys.argv[4], 'w'), indent=1)
