# kac-cosmo — Kinetic decoupling in an expanding universe as a geometric balance

Code and data accompanying

> T. A. Tazhimov, A. R. Sharden, V. S. Zheltukhin,
> *Kinetic decoupling in an expanding universe as a geometric balance between expansion and collisions*, submitted to Physical Review D (2026).

Python (NumPy, SciPy, Matplotlib). Run every script from inside `scripts/`; outputs go to `../data/` and `../figures/`.

## Scripts

| Script | Content | Paper |
|---|---|---|
| `kac_geom.py` | shared library: grid, Kac collision maps, collision operator | — |
| `flrw.py` | one-dimensional Kac model with relativistic kinematics in FLRW: forcing function F²(T), mode forcing F_n, three expanding runs (Γ₀ = 4, 40, 400), quasi-stationary prediction, frozen profiles | Sec. 4, Fig. 1, Table 1 |
| `snapshots.py` | re-runs `flrw` (Γ₀=4) and `wimp` (p=5) saving distribution snapshots | Figs. "evolution of the distribution" |
| `figsplit.py` | the two profile figures | Figs. 2, 5 |
| `reldsmc.py` | 3D relativistic hard-sphere gas (Møller velocity, CM-isotropic scattering, Lorentz boosts), DSMC/NTC; modes: `forcing`, `static` (contraction rate γ_S), `expand` (n₀σ = 3, 30, 300) | Sec. 5, Fig. 3, Table 2 |
| `figrel.py` | assembles Fig. 3 from `rel_*.json` | Fig. 3 |
| `wimp.py` | heavy relic + cooling bath: Kac thermostat with μ ∝ T_b^p, exact temperature ODE, full kinetic solution with shape (library + CLI) | Sec. 6 |
| `wimp4.py` | the run used in the paper: p = 4, μ0 = 21.5, Γ0 = 5 → `wimp_p4_mu21.json` | Table 3 |
| `figw4.py` | Fig. 5 (T_χ/T_b, δ vs drift-corrected law, shape) and Fig. 4 (profiles, p = 4); ODE for p = 4, 6 | Figs. 4–5 |
| `figflrw_en.py` | Fig. 1 from saved data | Fig. 1 |

## Reproduce
```bash
pip install numpy scipy matplotlib
cd scripts
python flrw.py                       # ≈ 15 min (three runs)
python figflrw_en.py
python reldsmc.py forcing            # 1 min
python reldsmc.py static 7 400000 0.3,0.7,1.5     # ≈ 5 min
python reldsmc.py expand 2 200000 30 140          # ≈ 5 min   (seed N  n0σ  t_end)
python reldsmc.py expand 3 200000 3 140
python reldsmc.py expand 8 1000000 300 140        # ≈ 15 min
python figrel.py
python wimp4.py 4 21.5 5             # ≈ 15 min  (p  μ0  Γ0)
python snapshots4.py wimp
python figw4.py
python snapshots.py wimp && python snapshots.py flrw && python figsplit.py
```
`data/` contains the outputs of the original runs so every table can be checked without recomputation.

## License
MIT. Please cite the paper and the Zenodo DOI of this repository.
