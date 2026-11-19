"""Validazione contro risultati analitici e semi-analitici. Uso: python validation.py (oppure pytest)."""
import numpy as np
from scipy import stats

from analysis import cumulative_incidence, pareto_mask
from model import (TARGET, Environment, FokkerPlanck, baseline, first_passage, load_config,
                   make_environment)

CFG = load_config()
ALL_KINETICS = ("force", "binding", "sinks")


def analytic_survival(t, D, L):
    """Primo passaggio 1D, parete riflettente in 0, assorbente in L, partenza in 0 (serie di Fourier)."""
    n = np.arange(200)[:, None]
    return (4 / np.pi * (-1.0) ** n / (2 * n + 1)
            * np.exp(-(2 * n + 1) ** 2 * np.pi ** 2 * D * np.asarray(t) / (4 * L ** 2))).sum(axis=0)


def params(homogeneous=False, on=(), **override):
    """Parametri nominali; il caso omogeneo ha D = 100 µm²/s."""
    return baseline(CFG, homogeneous, on, **{**({"D_bulk_um2_s": 100.0} if homogeneous else {}), **override})


def fokker_planck(p):
    return FokkerPlanck(Environment(p, CFG), CFG["integration"]["fp_cells_per_width"])


def frozen_orientation_hit(p, t):
    """P_hit(t) con orientazione congelata (D_R = 0): media su θ0 uniforme dei Fokker–Planck con D -> D(1 + a cos2θ0)."""
    thetas = np.linspace(0, np.pi, 16, endpoint=False)
    return np.mean([fokker_planck({**p, "D_bulk_um2_s": p["D_bulk_um2_s"] * (1 + p["anisotropy"] * np.cos(2 * th))})
                    .outcomes(t)[TARGET] for th in thetas], axis=0)


def dkw(n, alpha=0.01):
    """Scarto massimo atteso tra CDF empirica e vera (Dvoretzky–Kiefer–Wolfowitz)."""
    return np.sqrt(np.log(2 / alpha) / (2 * n))


def check_outcomes(p, p_ref, t_max, n, seed):
    """Cumulate di esito di Langevin (parametri p, anche 2D) contro Fokker–Planck 1D (parametri p_ref)."""
    T, outcome, _ = first_passage(make_environment(p, CFG), n, t_max, np.random.default_rng(seed))
    t = np.linspace(0, t_max, 100)
    ref = fokker_planck(p_ref).outcomes(t)
    for k, inc in cumulative_incidence(T, outcome, t).items():
        err = np.abs(inc - ref[k]).max()
        assert err < dkw(n), f"esito {k}: scarto {err:.4f} > DKW {dkw(n):.4f}"


def test_free_diffusion():
    """Diffusione libera: x(t) ~ N(x0, 2Dt) lontano dai bordi."""
    env, rng, n = Environment(params(True), CFG), np.random.default_rng(1), 4000
    x0, t = env.xT / 2, (env.xT / 16) ** 2 / (2 * 100.0)        # sigma = L/16: bordi a 8 sigma, qualunque sia la geometria
    _, _, x = first_passage(env, n, t, rng, x0=x0)
    p = stats.kstest(x[~np.isnan(x)], "norm", args=(x0, np.sqrt(2 * 100.0 * t))).pvalue
    assert p > 0.01, f"KS p-value = {p:.3g}"


def test_first_passage_homogeneous():
    """Primo passaggio omogeneo: CDF di T_hit simulata contro la serie analitica."""
    env, n, t_max = Environment(params(True), CFG), 3000, 2.0
    T, outcome, _ = first_passage(env, n, t_max, np.random.default_rng(2))
    t = np.linspace(0, t_max, 200)
    err = np.abs(cumulative_incidence(T, outcome, t)[TARGET] - (1 - analytic_survival(t, 100.0, env.xT))).max()
    assert err < dkw(n), f"scarto massimo {err:.4f} > DKW {dkw(n):.4f}"


def test_fokker_planck_vs_langevin():
    """Ambiente eterogeneo: P_hit(t) e densità dei sopravvissuti (SDE) contro Fokker–Planck."""
    p, n, t_max = params(), 3000, 1.0
    env, fp = Environment(p, CFG), fokker_planck(p)
    T, outcome, x_end = first_passage(env, n, t_max, np.random.default_rng(3))
    t = np.linspace(0, t_max, 100)
    err_t = np.abs(cumulative_incidence(T, outcome, t)[TARGET] - (1 - fp.survival(t))).max()
    assert err_t < dkw(n), f"P_hit(t): scarto {err_t:.4f} > DKW {dkw(n):.4f}"

    x = np.sort(x_end[outcome != TARGET])                        # CDF delle posizioni dei sopravvissuti
    edges = (np.arange(fp.n) + 1) * fp.h
    F_fp = np.cumsum(fp.density(t_max)) * fp.h / fp.survival(t_max)[0]
    err_x = np.abs(np.searchsorted(x, edges) / x.size - F_fp).max()
    assert err_x < dkw(x.size) + 0.01, f"densità: scarto {err_x:.4f} > DKW {dkw(x.size):.4f}"


def test_zero_barrier():
    """Senza contrasti di D né partizione il Fokker–Planck eterogeneo riduce alla soluzione omogenea."""
    t, L = np.geomspace(0.01, 5, 100), Environment(params(True), CFG).xT
    err = np.abs(fokker_planck(params(True)).survival(t) - analytic_survival(t, 100.0, L)).max()
    assert err < 5e-3, f"scarto {err:.2e}"


def test_perfect_barrier():
    """Membrana impermeabile (D -> 0): nessun attraversamento, né in FP né in Langevin."""
    p = params(**{"membrane.D_factor": 1e-12})
    assert fokker_planck(p).outcomes(CFG["t_max_s"])[TARGET][0] < 1e-6
    _, outcome, _ = first_passage(Environment(p, CFG), 300, 2.0, np.random.default_rng(4))
    assert not (outcome == TARGET).any()


def test_small_noise():
    """Limite di piccolo rumore: D -> 0 con v = D f fissata, T_hit -> L/v con dispersione ~ sqrt(2 D L / v³)."""
    cfg = {**CFG, "integration": {**CFG["integration"], "step_max_um": 0.05}}
    v, D = 1.0, 1e-3
    env = Environment(params(True, D_bulk_um2_s=D, force_per_um=v / D), cfg)
    T, outcome, _ = first_passage(env, 300, 12.0, np.random.default_rng(5))
    assert (outcome == TARGET).all() and abs(T.mean() - env.xT / v) < 0.02 * env.xT / v, f"T medio {T.mean():.3f}"
    assert T.std() < 0.5, f"dispersione {T.std():.3f}"


def test_outcomes_conservation():
    """P_target + P_off-target + P_clearance + S(t) = 1 con binding, pozzi e forza; senza pozzi P_target = 1 - S."""
    t = np.geomspace(1e-3, 50, 50)
    fp = fokker_planck(params(on=ALL_KINETICS))
    assert np.allclose(fp.survival(t) + sum(fp.outcomes(t).values()), 1, atol=1e-6)
    fp = fokker_planck(params())
    assert np.allclose(fp.outcomes(t)[TARGET], 1 - fp.survival(t), atol=1e-6)


def test_binding_limit():
    """k_unbind -> infinito: frazione legata -> 0, si recupera il caso senza binding."""
    t = np.geomspace(0.01, 10, 50)
    bound = fokker_planck(params(on=("binding",), k_unbind=1e4)).outcomes(t)[TARGET]
    assert np.abs(bound - fokker_planck(params()).outcomes(t)[TARGET]).max() < 2e-3


def test_kinetics_vs_langevin():
    """Binding, clearance, off-target e forza esterna: cumulate di esito di Langevin contro Fokker–Planck."""
    p = params(on=ALL_KINETICS, **{"blood.k_clear": 3.0, "ecm.k_offtarget": 1.0, "cytoplasm.k_bind": 30.0, "k_unbind": 5.0})
    check_outcomes(p, p, t_max=2.0, n=1500, seed=6)       # esiti attesi a 2 s: ~0.67 / 0.13 / 0.13


def test_pores_limits():
    """2D: poro su tutta la barriera = barriera assente; porosità nulla = barriera intatta (1D)."""
    open_ref = params(**{"endothelium.D_factor": params()["blood.D_factor"]})
    check_outcomes(params(on=("pores",), porosity=1.0), open_ref, t_max=1.0, n=1000, seed=7)
    check_outcomes(params(on=("pores",), porosity=0.0), params(), t_max=1.0, n=1000, seed=8)


def test_anisotropy_limits():
    """Anisotropia: rotazione veloce -> D isotropo medio (FP 1D); rotazione congelata -> miscela su θ0."""
    n, t_max, t = 1000, 1.0, np.linspace(0, 1.0, 100)
    p = params(on=("anisotropy",), anisotropy=0.8)
    check_outcomes({**p, "D_R_per_s": 1e3}, params(), t_max, n, seed=9)
    T, outcome, _ = first_passage(make_environment({**p, "D_R_per_s": 0.0}, CFG), n, t_max, np.random.default_rng(10))
    err = np.abs(cumulative_incidence(T, outcome, t)[TARGET] - frozen_orientation_hit(p, t)).max()
    assert err < dkw(n), f"orientazione congelata: scarto {err:.4f} > DKW {dkw(n):.4f}"


def test_pareto_mask():
    F = np.array([[1, 2], [2, 1], [2, 2], [0.5, 3]])
    assert pareto_mask(F).tolist() == [True, True, False, True]

# ---------------------------------------------------------------------------
# Casi limite aggiuntivi: gaussiana libera, barriera perfetta, piccolo rumore
# ---------------------------------------------------------------------------

def gaussian_density_reflecting(x, t, D, x0=0.0):
    """Densità analitica di diffusione libera 1D con bordo riflettente in x=0.

    Metodo delle immagini:
        P(x,t | x0) = G(x - x0; t) + G(x + x0; t)
    con G(y;t) = exp(-y^2 / (4 D t)) / sqrt(4 pi D t).

    Se x0 = 0 (partenza sul bordo riflettente), P(x,t) = 2 G(x;t) per x > 0.
    Usata per validare il motore Langevin in assenza di drift e barriere.
    """
    x = np.asarray(x, dtype=float)
    G = lambda y: np.exp(-y**2 / (4.0 * D * t)) / np.sqrt(4.0 * np.pi * D * t)
    return G(x - x0) + G(x + x0)


def perfect_barrier_P_hit(cfg, rng, n_traj, t_max, barrier_name,
                          D_floor=1e-6, seed_offset=0):
    """P_hit empirica con D_factor -> 0 in un compartimento barriera.

    Con D_floor << 1 la barriera è praticamente impermeabile: P_hit deve crollare
    a ~0 entro l'errore MC.  Ritorna (P_hit, SE_MC, n_hit, n_total).
    Import locali per non creare dipendenze a livello di modulo da model.
    """
    from model import baseline, Environment, first_passage_parallel, TARGET
    p = baseline(cfg, **{f"{barrier_name}.D_factor": D_floor})
    env = Environment(p, cfg)
    # seed derivato per non collidere con le simulazioni scientifiche
    rng_local = np.random.default_rng([cfg["seed"], seed_offset, ord("V")])
    _, outcome, _ = first_passage_parallel(env, n_traj, t_max, rng_local)
    n_hit = int((outcome == TARGET).sum())
    P = n_hit / n_traj
    se = np.sqrt(P * (1.0 - P) / n_traj) if n_traj > 0 else 0.0
    return P, se, n_hit, n_traj


def small_noise_mean_trajectory(cfg, rng, D_values, v_target,
                                barrier_name=None, seed_offset=1):
    """Limite di piccolo rumore: con drift uniforme di velocità fissata v_target,
    al diminuire di D la traiettoria media deve convergere a x(t) = v_target * t.

    In questo modello il drift è b(x) = mu * F = (D / kT) * F.  Per mantenere
    v_target costante mentre D varia, si impone F = v_target * kT / D.
    Ritorna un dict {D: (t_grid, <x(t)>_empirico)}.
    """
    from model import baseline, Environment, first_passage_parallel, TARGET
    kT = 1.0  # unità interne: kT = 1, forza in kT/µm
    out = {}
    for D in D_values:
        F = v_target * kT / D
        kwargs = {"force_per_um": F}
        if barrier_name is not None:
            kwargs[f"{barrier_name}.D_factor"] = 1e-6
        p = baseline(cfg, on=("force",), **kwargs)
        env = Environment(p, cfg)
        rng_local = np.random.default_rng([cfg["seed"], seed_offset, int(D * 1e6)])
        # raccogli la traiettoria media su una griglia temporale grossolana
        t_end = min(cfg["t_max_s"], 0.6 * env.xT / max(v_target, 1e-12))
        n = 200
        X = np.empty((n,))
        t_grid = np.linspace(0.0, t_end, n)
        # campiona traiettorie e interpola su t_grid (poche traiettorie, sufficiente)
        n_show = min(200, cfg["n_traj"])
        acc = np.zeros_like(t_grid)
        for _ in range(n_show):
            x, t, aux, path = np.zeros(1), np.zeros(1), env.init_aux(1, rng_local), [(0.0, 0.0)]
            while t[0] < t_end and x[0] < env.xT:
                x, aux, t, _ = env.step(x, aux, t, t_end, rng_local)
                path.append((t[0], x[0]))
            path = np.array(path)
            acc += np.interp(t_grid, path[:, 0], path[:, 1])
        out[D] = (t_grid, acc / n_show)
    return out


if __name__ == "__main__":
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"PASS {name}", flush=True)