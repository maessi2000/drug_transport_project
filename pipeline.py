"""Esperimenti A–H. Uso: python pipeline.py [A B C ...]   (default: tutti; risultati in results/<esperimento>/)
A omogeneo | B eterogeneo (D(x), interfacce, partizione) | C molecole reali (data/molecules.csv) | D descrittori
E regimi adimensionali (Pe, Da) | F esiti: bersaglio / off-target / clearance | G fronte di Pareto | H 2D e anisotropia
V validazione estesa: diffusione libera, barriera perfetta, piccolo rumore"""

import json
import sys
import warnings
from dataclasses import asdict

import matplotlib
import numpy as np
import pandas as pd
import scipy

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from analysis import cox_hazard_ratios, cumulative_incidence, kaplan_meier, pareto_mask, propagate, sensitivity
import model
from model import (OUTCOMES, ROOT, TARGET, CENSORED, Environment, FokkerPlanck, baseline, configure, env_params,
                   extension_params, first_passage_parallel, load_config, load_molecules, make_environment, nominal,
                   reference_params, simulate, transport_summary, transport_summaries)
from validation import (analytic_survival, frozen_orientation_hit,
                        gaussian_density_reflecting, perfect_barrier_P_hit,
                        small_noise_mean_trajectory)


# ---------------------------------------------------------------- utilità
def setup(exp, cfg):
    """Cartella dei risultati e generatore casuale riproducibile (seed di configurazione + esperimento)."""
    out = ROOT / "results" / exp
    out.mkdir(parents=True, exist_ok=True)
    return out, np.random.default_rng([cfg["seed"], ord(exp)])


def record(out, cfg, **extra):
    """Registro di riproducibilità: configurazione completa (seed, passo, t_max, N, interfacce, provenienza)
    più valori effettivi usati, numero di processi (i flussi casuali dipendono da seed e workers) e versioni."""
    meta = {"config": cfg, "workers": model.WORKERS, "numpy": np.__version__, "scipy": scipy.__version__, **extra}
    (out / "run.json").write_text(json.dumps(meta, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def langevin(env, cfg, rng):
    """Traiettorie con N fisso oppure precisione MC richiesta; avvisa se la censura è eccessiva."""
    T, outcome, history = simulate(env, rng, cfg["t_max_s"], n=None if cfg["target_se"] else cfg["n_traj"],
                                   se=cfg["target_se"])
    censored = (outcome == CENSORED).mean()
    if censored > cfg["max_censoring"]:
        warnings.warn(f"frazione censurata {censored:.0%} > {cfg['max_censoring']:.0%}: "
                      "aumentare t_max_s o interpretare P_hit con cautela")
    return T, outcome, history


def sample_path(env, rng, t_end):
    x, t, aux, path = np.zeros(1), np.zeros(1), env.init_aux(1, rng), [(0.0, 0.0)]
    while t[0] < t_end and x[0] < env.xT:
        x, aux, t, _ = env.step(x, aux, t, t_end, rng)
        path.append((t[0], x[0]))
    return np.array(path)


def shade_compartments(ax, env, cfg, horizontal=True):
    """Colora i compartimenti (bande orizzontali se x è sull'asse y, verticali altrimenti) e li etichetta."""
    bounds = [0, *env.interfaces, env.xT]
    for i, (lo, hi, c) in enumerate(zip(bounds[:-1], bounds[1:], cfg["environment"]["compartments"])):
        if horizontal:
            ax.axhspan(lo, hi, color=f"C{i}", alpha=0.15, lw=0)
            ax.text(0.99, (lo + hi) / 2, c["name"], transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)
        else:
            ax.axvspan(lo, hi, color=f"C{i}", alpha=0.15, lw=0)
            ax.text((lo + hi) / 2, 0.99, c["name"], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)


# ---------------------------------------------------------------- A: diffusione omogenea
def exp_A(cfg):
    out, rng = setup("A", cfg)
    p = baseline(cfg, homogeneous=True)
    env = Environment(p, cfg)
    T, outcome, hist = langevin(env, cfg, rng)
    event = outcome == TARGET
    km, _ = kaplan_meier(T, event)
    km.to_csv(out / "km_curve.csv")
    D, L = p["D_bulk_um2_s"], env.xT
    print(f"A: P_hit = {hist[-1][1]:.4f} ± {hist[-1][2]:.4f} (N={hist[-1][0]}), "
          f"T_hit medio = {T[event].mean():.3f} s, analitico L²/2D = {L**2 / (2 * D):.3f} s")
    t = np.linspace(0, np.quantile(T, 0.99), 300)
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    ax.fill_between(km.index, km.low, km.high, step="post", alpha=0.3)
    ax.step(km.index, km.P_hit, where="post", label="Kaplan–Meier (Langevin)")
    ax.plot(t, 1 - analytic_survival(t, D, L), "k--", label="analytic")
    ax.set(xlim=(0, t[-1]), xlabel="t (s)", ylabel="P_hit(t)", title="A: homogeneous diffusion")
    ax.legend()
    fig.savefig(out / "A_first_passage.png", dpi=150)
    record(out, cfg, parameters=p, convergence=hist)


# ---------------------------------------------------------------- B: ambiente eterogeneo
def exp_B(cfg):
    out, rng = setup("B", cfg)
    p = baseline(cfg)
    env = Environment(p, cfg)
    fp = FokkerPlanck(env, cfg["integration"]["fp_cells_per_width"])
    T, outcome, hist = langevin(env, cfg, rng)
    event = outcome == TARGET
    km, med = kaplan_meier(T, event)
    km.to_csv(out / "km_curve.csv")
    summary = transport_summary(p, cfg)
    print(f"B: P_hit = {hist[-1][1]:.4f} ± {hist[-1][2]:.4f} (FP: {summary['P_hit']:.4f}), median T_hit = {med:.3f} s "
          f"(FP: {summary['T_med']:.3f}), tau_D = {summary['tau_D']:.3f} s, T_med/tau_D = {med / summary['tau_D']:.2f}")

    t_star = med if np.isfinite(med) else cfg["t_max_s"] / 10        # istante per il confronto delle densità
    _, out_s, x_end = first_passage_parallel(env, cfg["n_traj"], t_star, rng)
    ev_s = out_s == TARGET
    t, t_hi = np.linspace(0, np.quantile(T, 0.99), 300), np.quantile(T, 0.99)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for _ in range(5):
        ax[0, 0].plot(*sample_path(env, rng, 3 * t_star).T, lw=0.8)
    shade_compartments(ax[0, 0], env, cfg)
    ax[0, 0].set(xlabel="t (s)", ylabel="x (µm)", title="compartments and sample trajectories")

    ax[0, 1].fill_between(km.index, km.low, km.high, step="post", alpha=0.3)
    ax[0, 1].step(km.index, km.P_hit, where="post", label="Kaplan–Meier (Langevin)")
    ax[0, 1].plot(t, 1 - fp.survival(t), "k--", label="Fokker–Planck")
    ax[0, 1].set(xlim=(0, t_hi), xlabel="t (s)", ylabel="P_hit(t)", title="cumulative hitting probability")
    ax[0, 1].legend()

    nb = 60
    ax[1, 0].hist(T[event], bins=nb, range=(0, t_hi), weights=np.full(event.sum(), nb / (T.size * t_hi)),
                  alpha=0.5, label="Langevin")
    ax[1, 0].plot(t, fp.hit_density(t), "k--", label="Fokker–Planck")
    ax[1, 0].set(xlabel="T_hit (s)", ylabel="first-passage density (1/s)", title="first-passage-time distribution")
    ax[1, 0].legend()

    alive = ~ev_s
    ax[1, 1].hist(x_end[alive], bins=40, range=(0, env.xT), weights=np.full(alive.sum(), 40 / (ev_s.size * env.xT)),
                  alpha=0.5, label="Langevin")
    ax[1, 1].plot(fp.x, fp.density(t_star), "k--", label="Fokker–Planck")
    shade_compartments(ax[1, 1], env, cfg, horizontal=False)
    ax[1, 1].set(xlabel="x (µm)", ylabel="P(x, t*)", title=f"density of surviving particles at t* = {t_star:.2f} s")
    ax[1, 1].legend()
    fig.savefig(out / "B_heterogeneous.png", dpi=150)
    record(out, cfg, parameters=p, convergence=hist, fokker_planck=summary, t_star=t_star)


# ---------------------------------------------------------------- C: molecole reali
def exp_C(cfg):
    import time

    out, rng = setup("C", cfg)
    rows, frames, curves, used = [], [], {}, {}

    molecules = list(load_molecules().items())
    n_molecules = len(molecules)
    t_exp = time.perf_counter()

    print(f"C: avvio simulazione di {n_molecules} molecole", flush=True)

    for i, (name, mol) in enumerate(molecules, 1):
        t_mol = time.perf_counter()
        print(f"\n  [{i}/{n_molecules}] {name}", flush=True)
        print("      • costruzione ambiente e simulazione Langevin...", flush=True)

        P = {**env_params(cfg), **mol}
        p = nominal(P)
        env = Environment(p, cfg)
        T, outcome, hist = langevin(env, cfg, rng)
        event = outcome == TARGET
        km, med = kaplan_meier(T, event)
        med = med if np.isfinite(med) else np.nan                      # mediana KM non raggiunta entro t_max

        print(f"      • Langevin completata: N={hist[-1][0]}, "
              f"P_hit={hist[-1][1]:.4f} ± {hist[-1][2]:.4f}, "
              f"mediana T_hit={med:.3f} s", flush=True)
        print("      • propagazione dell'incertezza dei parametri...", flush=True)

        _, Y = propagate(P, cfg, rng, cfg["uncertainty"]["n_samples"])      # incertezza dei parametri (FP)
        fp = transport_summary(p, cfg)

        censoring = float((outcome == CENSORED).mean())
        delta_P = float(hist[-1][1] - fp["P_hit"])
        rel_delta_P = delta_P / max(abs(fp["P_hit"]), 1e-12)
        T_cond_KM = float(np.median(T[event])) if event.any() else np.nan

        rows.append({"molecule": name, "D_bulk_um2_s": env.D_bulk, "D_kind": mol.get("D_bulk_um2_s").kind if "D_bulk_um2_s" in mol else "assumed",
                     "logP": p["logP"], "N": hist[-1][0],
                     "P_hit": hist[-1][1], "P_hit_SE_MC": hist[-1][2],
                     "P_hit_FP": fp["P_hit"], "P_hit_param_2.5%": Y.P_hit.quantile(0.025),
                     "P_hit_param_97.5%": Y.P_hit.quantile(0.975), "T_med_KM": med, "T_med_FP": fp["T_med"],
                     "tau_D": fp["tau_D"], "T_med_over_tau_D": med / fp["tau_D"],
                     "censoring": censoring, "Delta_P_Langevin_FP": delta_P, "Delta_P_over_FP": rel_delta_P, "T_cond_KM": T_cond_KM, "T_cond_FP": fp["T_cond"],
                     })
        frames.append(pd.DataFrame({"molecule": name, "T": T, "event": event, "log10_D": np.log10(env.D_bulk),
                                    "logP": p["logP"]}))
        curves[name], used[name] = km, {k: asdict(v) for k, v in mol.items()}

        elapsed_mol = time.perf_counter() - t_mol
        print(f"      ✓ completata in {elapsed_mol:.1f} s | "
              f"P_hit FP={fp['P_hit']:.4f}, T_med FP={fp['T_med']:.3f} s", flush=True)

    print(f"\nC: tutte le {n_molecules} molecole completate in "
          f"{time.perf_counter() - t_exp:.1f} s. Generazione del summary...", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out / "molecules_summary.csv", index=False)
    print(table.round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for name, km in curves.items():
        ax[0].step(km.index, km.P_hit, where="post", label=name)
    ax[0].set(xlabel="t (s)", ylabel="P_hit(t)", title="Kaplan–Meier: cumulative hitting probability")
    ax[0].legend(fontsize=8)
    cox = None
    if len(table) > 2:                       # con meno molecole le covariate non sono identificabili
        cox = cox_hazard_ratios(pd.concat(frames), ["log10_D", "logP"])
        cox.to_csv(out / "cox_hazard_ratios.csv")
        print(cox.round(4).to_string())
        if (cox.p_PH < 0.05).any():
            warnings.warn("rischi proporzionali violati per: " + ", ".join(cox.index[cox.p_PH < 0.05])
                          + ": gli HR non sono interpretabili come effetto costante nel tempo")
        y = np.arange(len(cox))
        ax[1].errorbar(cox.HR, y, xerr=[cox.HR - cox.low, cox.high - cox.HR], fmt="o")
        ax[1].set(yticks=y, yticklabels=[f"{i} (PH test p={p:.2f})" for i, p in zip(cox.index, cox.p_PH)],
                  xscale="log", xlabel="hazard ratio per +1 SD", title="Cox (descriptive, not causal)")
        ax[1].axvline(1, color="k", lw=0.8)
    else:
        ax[1].axis("off")
    fig.savefig(out / "C_molecules.png", dpi=150)
    record(out, cfg, molecules=used, summary=rows, cox=None if cox is None else cox.to_dict())

# ---------------------------------------------------------------- D: spazio dei descrittori
def exp_D(cfg):
    out, rng = setup("D", cfg)
    # il riferimento fissa D_bulk direttamente: T e viscosità non hanno effetto e sono esclusi dall'incertezza
    P = {k: v for k, v in {**env_params(cfg), **reference_params(cfg)}.items()
         if k not in ("temperature_K", "viscosity_Pa_s")}
    n = cfg["sweeps"]["n"]
    Dg = P["D_bulk_um2_s"].grid(n)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, key in enumerate(["logP", "membrane.D_factor"]):
        yv = P[key].grid(n)
        grid = [{"D_bulk_um2_s": d, key: v} for d in Dg for v in yv]
        res = pd.DataFrame(grid).join(pd.DataFrame(transport_summaries([baseline(cfg, **g) for g in grid], cfg)))
        res["T_med_over_tau_D"] = res.T_med / res.tau_D
        res.to_csv(out / f"sweep_{key}.csv", index=False)
        for col, (z, title, norm) in enumerate([("P_hit", f"P_hit(T = {cfg['t_max_s']:g} s)", None),
                                                ("T_med_over_tau_D", "median T_hit / tau_D", LogNorm())]):
            m = ax[row, col].pcolormesh(Dg, yv, res.pivot(index=key, columns="D_bulk_um2_s", values=z).values,
                                        shading="nearest", norm=norm)
            ax[row, col].set(xscale="log", yscale="log" if P[key].low > 0 else "linear", title=title,
                             xlabel="D_bulk (µm²/s)", ylabel=key)
            fig.colorbar(m, ax=ax[row, col])
    fig.savefig(out / "D_sweeps.png", dpi=150)

    X, Y = propagate(P, cfg, rng, cfg["uncertainty"]["n_samples"])
    X.join(Y).to_csv(out / "uncertainty_samples.csv", index=False)
    sens = sensitivity(X, Y)
    sens.to_csv(out / "sensitivity_spearman.csv")
    print(f"D: P_hit 95% interval = [{Y.P_hit.quantile(0.025):.3f}, {Y.P_hit.quantile(0.975):.3f}], "
          f"median T_hit 95% interval = [{Y.T_med.quantile(0.025):.3f}, {Y.T_med.quantile(0.975):.3f}] s")
    print(sens.round(2).to_string())

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    ax[0].hist(Y.P_hit, bins=30)
    ax[0].set(xlabel=f"P_hit(T = {cfg['t_max_s']:g} s)", title="uncertainty from parameter ranges")
    ax[1].hist(Y.T_med.dropna(), bins=30)
    ax[1].set(xlabel="median T_hit (s)", title="uncertainty from parameter ranges")
    sens.loc[sens.abs().max(axis=1).sort_values().index].plot.barh(ax=ax[2])
    ax[2].set(xlabel="Spearman rank correlation", title="sensitivity")
    fig.savefig(out / "D_uncertainty_sensitivity.png", dpi=150)
    record(out, cfg, parameters={k: asdict(v) for k, v in P.items()})


# ---------------------------------------------------------------- E: regimi adimensionali
def exp_E(cfg):
    """Advezione (Pe = f L) e binding (Da = k_bind τ_D): P_hit e T_cond/τ_D da Fokker–Planck."""
    out, _ = setup("E", cfg)
    P, n = {**extension_params(cfg, ("force", "binding"))}, cfg["sweeps"]["n"]
    force = pd.DataFrame(transport_summaries([baseline(cfg, force_per_um=f) for f in P["force_per_um"].grid(n)], cfg))
    combos = [(ku, kb) for ku in P["k_unbind"].grid(3) for kb in P["cytoplasm.k_bind"].grid(n)]
    bind = pd.DataFrame(transport_summaries([baseline(cfg, on=("binding",), **{"cytoplasm.k_bind": kb, "k_unbind": ku})
                                             for ku, kb in combos], cfg)).assign(k_unbind=[ku for ku, _ in combos])
    ref = transport_summary(baseline(cfg), cfg)
    for df in (force, bind):
        df["T_over_tau_D"] = df.T_cond / df.tau_D
    force.to_csv(out / "sweep_force.csv", index=False)
    bind.to_csv(out / "sweep_binding.csv", index=False)
    print(f"E: Pe = {force.Pe.iloc[-1]:.1f}: T_cond/tau_D = {force.T_over_tau_D.iloc[-1]:.4f} (limite avvettivo 1/Pe = "
          f"{1 / force.Pe.iloc[-1]:.4f}); senza forza e binding: T_cond/tau_D = {ref['T_cond'] / ref['tau_D']:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    ax[0].loglog(force.Pe, force.T_over_tau_D, "o-", label="Fokker–Planck")
    ax[0].loglog(force.Pe, 1 / force.Pe, "k--", label="advective limit 1/Pe")
    ax[0].axhline(ref["T_cond"] / ref["tau_D"], color="gray", lw=0.8, label="f = 0")
    ax[0].set(xlabel="Pe = f L", ylabel="median T_hit / tau_D", title="advection")
    ax[0].legend()
    for ku, g in bind.groupby("k_unbind"):
        ax[1].semilogx(g.Da, g.P_hit, "o-", label=f"k_unbind = {ku:.2g} 1/s")
        ax[2].loglog(g.Da, g.T_over_tau_D, "o-", label=f"k_unbind = {ku:.2g} 1/s")
    ax[1].set(xlabel="Da = k_bind tau_D", ylabel=f"P_hit(T = {cfg['t_max_s']:g} s)", title="binding: delivery probability")
    ax[2].set(xlabel="Da = k_bind tau_D", ylabel="median T_hit / tau_D", title="binding: arrival time")
    ax[1].legend()
    fig.savefig(out / "E_dimensionless_regimes.png", dpi=150)
    record(out, cfg, parameters={k: asdict(v) for k, v in P.items()})


# ---------------------------------------------------------------- F: esiti concorrenti
def exp_F(cfg):
    """Bersaglio, off-target, clearance e censura: cumulate empiriche (Langevin) contro Fokker–Planck."""
    out, rng = setup("F", cfg)
    p = baseline(cfg, on=("sinks",))
    env = Environment(p, cfg)
    T, outcome, hist = langevin(env, cfg, rng)
    t = np.linspace(0, cfg["t_max_s"], 300)
    inc = cumulative_incidence(T, outcome, t)
    ref = FokkerPlanck(env, cfg["integration"]["fp_cells_per_width"]).outcomes(t)
    inc[CENSORED], ref[CENSORED] = 1 - sum(inc.values()), 1 - sum(ref.values())
    table = pd.DataFrame([{"outcome": OUTCOMES[k], "P_MC": inc[k][-1], "SE_MC": np.sqrt(inc[k][-1] * (1 - inc[k][-1]) / outcome.size),
                           "P_FP": ref[k][-1]} for k in inc])
    table.to_csv(out / "outcomes.csv", index=False)
    print("F:\n" + table.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    for k in inc:
        ax.plot(t, inc[k], color=f"C{k}", label=OUTCOMES[k])
        ax.plot(t, ref[k], "--", color="k", lw=0.8)
    ax.plot([], [], "k--", lw=0.8, label="Fokker–Planck")
    ax.set(xlabel="t (s)", ylabel="cumulative probability of first outcome", title="F: competing outcomes")
    ax.legend()
    fig.savefig(out / "F_outcomes.png", dpi=150)
    record(out, cfg, parameters=p, convergence=hist)


# ---------------------------------------------------------------- G: Pareto
def exp_G(cfg):
    """Fronte di Pareto sui descrittori molecolari (D_bulk, logP): max P_hit, min T_cond, min P_offtarget."""
    out, rng = setup("G", cfg)
    design, base = reference_params(cfg), baseline(cfg, on=("binding", "sinks"))
    X = pd.DataFrame([{k: v.sample(rng) for k, v in design.items()} for _ in range(cfg["pareto"]["n_candidates"])])
    R = X.join(pd.DataFrame(transport_summaries([{**base, **row} for row in X.to_dict("records")], cfg)))
    R = R.dropna(subset=["T_cond"]).assign(pareto=lambda r: pareto_mask(np.c_[-r.P_hit, r.T_cond, r.P_offtarget]))
    R.to_csv(out / "candidates.csv", index=False)
    print(f"G: {R.pareto.sum()} candidati Pareto-ottimi su {len(R)}")
    print(R[R.pareto].sort_values("T_cond")[["D_bulk_um2_s", "logP", "P_hit", "T_cond", "P_offtarget"]].round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    sc = ax[0].scatter(R.T_cond, R.P_hit, c=R.P_offtarget, s=14)
    ax[0].scatter(R.T_cond[R.pareto], R.P_hit[R.pareto], facecolors="none", edgecolors="r", s=60, label="Pareto front")
    ax[0].set(xlabel="median arrival time T_cond (s)", ylabel=f"P_hit(T = {cfg['t_max_s']:g} s)", title="objective space")
    ax[0].legend()
    fig.colorbar(sc, ax=ax[0], label="P_off-target")
    ax[1].scatter(R.D_bulk_um2_s, R.logP, c=np.where(R.pareto, "r", "lightgray"), s=14)
    ax[1].set(xscale="log", xlabel="D_bulk (µm²/s)", ylabel="logP", title="descriptor space (red = Pareto-optimal)")
    fig.savefig(out / "G_pareto.png", dpi=150)
    record(out, cfg, base=base, design={k: asdict(v) for k, v in design.items()})


# ---------------------------------------------------------------- H: 2D e anisotropia
def exp_H(cfg):
    """Pori nella barriera (porosità) e anisotropia con rotazione (D_R): Langevin 2D contro i limiti di Fokker–Planck 1D.
    Costo ridotto: n_traj/3 traiettorie fino a t_max/2."""
    out, rng = setup("H", cfg)
    n, t_max = cfg["n_traj"] // 3, cfg["t_max_s"] / 2
    t, P = np.linspace(0, t_max, 200), extension_params(cfg, ("pores", "anisotropy"))
    fp_hit = lambda p: FokkerPlanck(Environment(p, cfg), cfg["integration"]["fp_cells_per_width"]).outcomes(t)[TARGET]

    def langevin_hit(p):
        T, outcome, _ = first_passage_parallel(make_environment(p, cfg), n, t_max, rng)
        return cumulative_incidence(T, outcome, t)[TARGET]

    names, E = [c["name"] for c in cfg["environment"]["compartments"]], env_params(cfg)
    k = cfg["geometry_2d"]["pore_compartment"]
    up = names[names.index(k) - 1]                                    # a monte: il poro ha il suo D
    # barriera all'estremo sfavorevole dell'intervallo plausibile, così che i pori abbiano un effetto misurabile
    strong = {f"{k}.D_factor": E[f"{k}.D_factor"].low, f"{k}.width": E[f"{k}.width"].high}
    base = baseline(cfg)
    tau = Environment(base, cfg).tau_D
    curves = {"intact barrier (FP 1D)": fp_hit(baseline(cfg, **strong)),
              "no barrier (FP 1D)": fp_hit(baseline(cfg, **{**strong, f"{k}.D_factor": base[f"{up}.D_factor"]}))}
    for por in (0.05, P["porosity"].value, 0.5):
        curves[f"pores, porosity {por:g}"] = langevin_hit(baseline(cfg, on=("pores",), porosity=por, **strong))
    pa = baseline(cfg, on=("anisotropy",))
    curves["isotropic = fast rotation (FP 1D)"] = fp_hit(base)
    curves["frozen orientation (FP mixture)"] = frozen_orientation_hit(pa, t)
    for D_R in (0.0, P["D_R_per_s"].value, P["D_R_per_s"].high):
        curves[f"anisotropic, D_R = {D_R:g}/s (D_R tau_D = {D_R * tau:.2g})"] = langevin_hit({**pa, "D_R_per_s": D_R})
    pd.DataFrame({"t": t, **curves}).to_csv(out / "hit_curves.csv", index=False)
    print("H: P_hit(t_max) =", {c: round(float(v[-1]), 3) for c, v in curves.items()})

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for a, keys, title in zip(ax, ([c for c in curves if "pores" in c or "barrier" in c],
                                   [c for c in curves if "anisotropic" in c or "frozen" in c or "isotropic =" in c]),
                              (f"2D: pores in the {k} (barrier at the unfavourable end of its range)",
                               f"anisotropy a = {pa['anisotropy']:g} and rotational diffusion")):
        for c in keys:
            a.plot(t, curves[c], "--" if c.startswith(("intact", "isotropic")) else ":" if "FP" in c else "-",
                   color="k" if "FP" in c else None, label=c)
        a.set(xlabel="t (s)", ylabel="P_hit(t)", title=title)
        a.legend(fontsize=8)
    fig.savefig(out / "H_2d_anisotropy.png", dpi=150)
    record(out, cfg, parameters={k: asdict(v) for k, v in P.items()}, n=n, t_max=t_max)

# ---------------------------------------------------------------- V: validazione estesa
def exp_V(cfg):
    """Casi limite mancanti: gaussiana libera, barriera perfetta, piccolo rumore.
    Non è un esperimento scientifico: verifica i limiti analitici del simulatore."""
    out, rng = setup("V", cfg)
    checks = {}

    # --- 1) diffusione libera: densità Langevin vs gaussiana riflettente ---
    p = baseline(cfg, homogeneous=True)
    env = Environment(p, cfg)
    D = p["D_bulk_um2_s"]
    # tempo breve: la particella non deve sentire il target assorbente né il bordo opposto
    t_star = 0.02 * env.xT**2 / D
    _, _, x_end = first_passage_parallel(env, cfg["n_traj"], t_star, rng)
    alive = x_end < env.xT                      # escludi le traiettorie già assorbite
    x = np.linspace(0.0, env.xT, 200)
    pdf = gaussian_density_reflecting(x, t_star, D, x0=0.0)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    ax[0].hist(x_end[alive], bins=60, range=(0, env.xT), density=True,
               alpha=0.5, label="Langevin (sopravvissute)")
    ax[0].plot(x, pdf, "k--", label="gaussiana riflettente (analitica)")
    ax[0].set(xlabel="x (µm)", ylabel="P(x, t*)", title=f"1) diffusione libera, t* = {t_star:.3g} s")
    ax[0].legend()

    # confronto quantitativo: istogramma vs densità analitica
    edges = np.linspace(0.0, env.xT, 61)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(x_end[alive], bins=edges, density=True)
    pdf_centers = gaussian_density_reflecting(centers, t_star, D, x0=0.0)
    diff = hist - pdf_centers

    checks["free_diffusion_max_abs_diff"] = float(np.max(np.abs(diff)))
    checks["free_diffusion_L1"] = float(np.sum(np.abs(diff) * np.diff(edges)))

    # --- 2) barriera perfetta: D_factor -> 0 in membrana, P_hit deve crollare ---
    P_bar, se_bar, n_hit, n_tot = perfect_barrier_P_hit(
        cfg, rng, cfg["n_traj"], cfg["t_max_s"], "membrane", D_floor=1e-6, seed_offset=2)
    P_ref = transport_summary(baseline(cfg), cfg)["P_hit"]
    checks["perfect_barrier_P_hit"] = P_bar
    checks["perfect_barrier_SE_MC"] = se_bar
    checks["reference_P_hit"] = P_ref
    ax[1].bar(["riferimento", "barriera perfetta"],
              [P_ref, P_bar], yerr=[0.0, se_bar], color=["C0", "C3"])
    ax[1].set(ylabel="P_hit", title="2) barriera perfetta (D_factor = 1e-6)")
    ax[1].axhline(0.0, color="k", lw=0.5)

    # --- 3) piccolo rumore: <x(t)> -> v t quando D -> 0 a v fissata ---
    v_target = 1.0                              # µm/s, velocità di drift imposta
    D_values = [100.0, 10.0, 1.0, 0.1]
    traj = small_noise_mean_trajectory(cfg, rng, D_values, v_target,
                                       seed_offset=3)
    for D in D_values:
        t_grid, x_mean = traj[D]
        ax[2].plot(t_grid, x_mean, label=f"D = {D:g} µm²/s")
    t_det = np.linspace(0.0, max(traj[D_values[-1]][0]), 100)
    ax[2].plot(t_det, v_target * t_det, "k--", label=f"deterministico v·t (v={v_target} µm/s)")
    ax[2].set(xlabel="t (s)", ylabel="<x(t)> (µm)", title="3) piccolo rumore: convergenza al deterministico")
    ax[2].legend(fontsize=8)
    fig.savefig(out / "V_validation.png", dpi=150)
    print(f"V: gaussiana max|diff| = {checks['free_diffusion_max_abs_diff']:.3g}, "
          f"P_hit barriera = {P_bar:.4f} ± {se_bar:.4f} (rif. {P_ref:.4f}), "
          f"<x> piccolo rumore vs v·t: vedi V_validation.png")
    record(out, cfg, checks=checks, D_values=D_values, v_target=v_target)

EXPERIMENTS = {"A": exp_A, "B": exp_B, "C": exp_C, "D": exp_D, "E": exp_E, "F": exp_F, "G": exp_G, "H": exp_H, "V": exp_V}

if __name__ == "__main__":
    cfg = load_config()
    configure(cfg)
    for name in sys.argv[1:] or EXPERIMENTS:
        print(f"\n== Experiment {name}", flush=True)
        EXPERIMENTS[name](cfg)