#!/usr/bin/env python3
"""
Diagnostica Langevin vs Fokker-Planck.
Da mettere accanto a model.py, config.yaml e data/molecules.csv.

Esempi:
    python diagnostics_transport.py --quick
    python diagnostics_transport.py --molecules Urea Saccarosio Atenololo Metotrexato Morfina
    python diagnostics_transport.py --full
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model
from model import CENSORED, TARGET, Environment, FokkerPlanck

ROOT = Path(__file__).resolve().parent


def cfg_copy(cfg, **kw):
    c = copy.deepcopy(cfg)
    for k, v in kw.items():
        if k == "t_max":
            c["t_max_s"] = float(v)
        elif k == "step_min":
            c["integration"]["step_min_um"] = float(v)
        elif k == "step_max":
            c["integration"]["step_max_um"] = float(v)
        elif k == "fp_cells":
            c["integration"]["fp_cells_per_width"] = int(v)
    return c


def mol(name, cfg):
    ds = model.load_molecules(ROOT / "data" / "molecules.csv")
    P = {**model.env_params(cfg), **ds[name]}
    return model.nominal(P)


def fp(p, cfg, cells):
    env = Environment(p, cfg)
    t = np.geomspace(max(cfg["t_max_s"] * 1e-5, 1e-5), cfg["t_max_s"], 500)
    out = FokkerPlanck(env, cells).outcomes(t)
    ph = float(out[TARGET][-1])
    return {
        "P": ph,
        "Tmed": float(np.interp(.5, out[TARGET], t)) if ph >= .5 else np.nan,
        "Tcond": float(np.interp(ph / 2, out[TARGET], t)) if ph > 0 else np.nan,
        "tau": float(env.tau_D),
    }


def lv(p, cfg, n, seed):
    env = Environment(p, cfg)
    T, o, _ = model.first_passage_parallel(
        env, n, cfg["t_max_s"], np.random.default_rng(seed)
    )
    hit = o == TARGET
    P = float(hit.mean())
    return {
        "P": P,
        "SE": math.sqrt(P * (1 - P) / n),
        "Tcond": float(np.median(T[hit])) if hit.any() else np.nan,
        "censor": float((o == CENSORED).mean()),
    }


def gaussian_check(cfg, n=4000, seed=1):
    c = cfg_copy(cfg)
    p = model.baseline(c, homogeneous=True)
    env = Environment(p, c)
    D = p["D_bulk_um2_s"]
    t = min(.005 * env.xT**2 / D, .02)

    _, o, x_end = model.first_passage_parallel(
        env, n, t, np.random.default_rng(seed)
    )
    alive = x_end < env.xT

    edges = np.linspace(0, env.xT, 61)
    centers = .5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(x_end[alive], bins=edges, density=True)
    pdf = 2 / np.sqrt(4 * np.pi * D * t) * np.exp(-centers**2 / (4 * D * t))
    dx = np.diff(edges)

    out = pd.DataFrame([{
        "N": n,
        "t_s": t,
        "alive_fraction": float(alive.mean()),
        "max_abs_diff": float(np.max(np.abs(hist - pdf))),
        "L1_diff": float(np.sum(np.abs(hist - pdf) * dx)),
    }])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(x_end[alive], bins=edges, density=True, alpha=.5, label="Langevin")
    ax.plot(centers, pdf, "k--", label="reflecting Gaussian")
    ax.set(xlabel="x (µm)", ylabel="P(x,t)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "DIAGNOSTICS_gaussian.png", dpi=150)
    plt.close(fig)
    return out


def fp_convergence(cfg, names, cells):
    rows = []
    for name in names:
        p = mol(name, cfg)
        vals = []
        for m in cells:
            r = fp(p, cfg, m)
            vals.append(r["P"])
            rows.append({
                "molecule": name,
                "fp_cells": m,
                "P_FP": r["P"],
                "Tmed_FP": r["Tmed"],
                "Tcond_FP": r["Tcond"],
            })
        ref = vals[-1]
        for row in rows[-len(cells):]:
            row["rel_to_finest"] = (row["P_FP"] - ref) / max(abs(ref), 1e-12)
    return pd.DataFrame(rows)


def langevin_convergence(cfg, names, steps, n, seeds=(10, 11, 12)):
    rows = []
    for name in names:
        p = mol(name, cfg)
        for h in steps:
            c = cfg_copy(
                cfg,
                step_min=h,
                step_max=min(cfg["integration"]["step_max_um"], 10 * h),
            )
            ref = fp(p, c, c["integration"]["fp_cells_per_width"])
            vals = []
            cs = []
            t0 = time.perf_counter()
            for seed in seeds:
                r = lv(p, c, n, seed)
                vals.append(r["P"])
                cs.append(r["censor"])
            vals = np.asarray(vals)
            rows.append({
                "molecule": name,
                "step_min": h,
                "N": n,
                "P_LV_mean": vals.mean(),
                "P_LV_sd_seed": vals.std(ddof=1) if len(vals) > 1 else 0.,
                "P_FP": ref["P"],
                "Delta": vals.mean() - ref["P"],
                "rel_Delta": (vals.mean() - ref["P"]) / max(abs(ref["P"]), 1e-12),
                "censor_mean": np.mean(cs),
                "elapsed_s": time.perf_counter() - t0,
            })
    return pd.DataFrame(rows)


def mc_stability(cfg, names, Ns=(1000, 3000, 6000), seeds=(20, 21, 22, 23, 24)):
    rows = []
    for name in names:
        p = mol(name, cfg)
        ref = fp(p, cfg, cfg["integration"]["fp_cells_per_width"])["P"]
        for n in Ns:
            vals = np.array([lv(p, cfg, n, s)["P"] for s in seeds])
            rows.append({
                "molecule": name,
                "N": n,
                "P_mean": vals.mean(),
                "P_sd_seed": vals.std(ddof=1),
                "P_FP": ref,
                "Delta": vals.mean() - ref,
            })
    return pd.DataFrame(rows)


def tmax_check(cfg, names, tmaxs=(2.5, 5, 10, 20), n=3000):
    rows = []
    for name in names:
        p = mol(name, cfg)
        for tmax in tmaxs:
            c = cfg_copy(cfg, t_max=tmax)
            f = fp(p, c, c["integration"]["fp_cells_per_width"])
            r = lv(p, c, n, 30)
            rows.append({
                "molecule": name,
                "tmax_s": tmax,
                "P_LV": r["P"],
                "SE": r["SE"],
                "P_FP": f["P"],
                "censoring": r["censor"],
                "Tcond_LV": r["Tcond"],
                "Tcond_FP": f["Tcond"],
            })
    return pd.DataFrame(rows)


def milstein_P(p, cfg, n, seed):
    env = Environment(p, cfg)
    tmax = cfg["t_max_s"]
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    t = np.zeros(n)
    idx = np.arange(n)
    hit = np.zeros(n, dtype=bool)

    while idx.size:
        D = np.interp(x, env.xg, env.D_tab)
        g = np.interp(x, env.xg, env.g_tab)
        xi = np.interp(x, env.xg, env.xi_tab)
        dt, last = env._timestep(x, D, g, xi, t, tmax)

        dW = np.sqrt(dt) * rng.standard_normal(idx.size)
        h = max(env.w / 100, 1e-6)
        xp = np.clip(x + h, 0, env.xT)
        xm = np.clip(x - h, 0, env.xT)
        dD = (env.profiles(xp)[0] - env.profiles(xm)[0]) / np.maximum(xp - xm, 1e-12)

        xold = x.copy()
        x = np.abs(x + D * g * dt + np.sqrt(2 * D) * dW
                   + .5 * dD * (dW*dW - dt))
        tnew = np.where(last, tmax, t + dt)

        p_cross = np.exp(
            -np.maximum((env.xT - xold) * (env.xT - x), 0)
            / np.maximum(D * dt, 1e-300)
        )
        this = (x >= env.xT) | (rng.random(idx.size) < p_cross)

        hit[idx[this]] = True
        idx = idx[~(this | (tnew >= tmax))]
        x = x[~(this | (tnew >= tmax))]
        t = tnew[~(this | (tnew >= tmax))]

    return float(hit.mean())


def compare_milstein(cfg, names, steps=(.05, .025, .0125), n=1500):
    rows = []
    for name in names:
        p = mol(name, cfg)
        for h in steps:
            c = cfg_copy(
                cfg, step_min=h,
                step_max=min(cfg["integration"]["step_max_um"], 10*h),
                t_max=min(cfg["t_max_s"], 2.)
            )
            Pfp = fp(p, c, c["integration"]["fp_cells_per_width"])["P"]
            Pe = lv(p, c, n, 40)["P"]
            Pm = milstein_P(p, c, n, 40)
            rows.append({
                "molecule": name,
                "step_min": h,
                "P_Euler": Pe,
                "P_Milstein": Pm,
                "P_FP": Pfp,
                "abs_err_Euler": abs(Pe - Pfp),
                "abs_err_Milstein": abs(Pm - Pfp),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--molecules", nargs="+",
        default=["Urea", "Saccarosio", "Atenololo", "Metotrexato", "Morfina"]
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    cfg = model.load_config(ROOT / "config.yaml")
    model.configure(cfg)

    if args.quick:
        cells = [2, 4, 8, 16]
        steps = [.10, .05, .025]
        nstep = 800
        Ns = [1000, 3000]
        tmaxs = [5., 10., 20.]
    else:
        cells = [2, 4, 8, 16, 32]
        steps = [.10, .05, .025, .0125]
        nstep = 1500
        Ns = [1000, 3000, 6000]
        tmaxs = [2.5, 5., 10., 20.]

    g = gaussian_check(cfg)
    f = fp_convergence(cfg, args.molecules, cells)
    l = langevin_convergence(cfg, args.molecules, steps, nstep)
    m = mc_stability(cfg, args.molecules, Ns)
    t = tmax_check(cfg, args.molecules, tmaxs, n=1500 if args.quick else 3000)

    g.to_csv(out / "DIAGNOSTICS_gaussian.csv", index=False)
    f.to_csv(out / "DIAGNOSTICS_fp_convergence.csv", index=False)
    l.to_csv(out / "DIAGNOSTICS_langevin_convergence.csv", index=False)
    m.to_csv(out / "DIAGNOSTICS_mc_stability.csv", index=False)
    t.to_csv(out / "DIAGNOSTICS_tmax.csv", index=False)

    if args.full:
        mi = compare_milstein(cfg, args.molecules)
        mi.to_csv(out / "DIAGNOSTICS_milstein.csv", index=False)
    else:
        mi = None

    # Riassunto semplice: quanto si avvicina il punto più fine a FP?
    rows = []
    for name in args.molecules:
        z = l[l.molecule == name].sort_values("step_min")
        if len(z):
            coarse = z.iloc[0]
            fine = z.iloc[-1]
            rows.append({
                "molecule": name,
                "P_FP_finest": float(fine.P_FP),
                "P_Euler_coarse": float(coarse.P_LV_mean),
                "P_Euler_fine": float(fine.P_LV_mean),
                "Delta_coarse": float(coarse.Delta),
                "Delta_fine": float(fine.Delta),
                "error_reduced": abs(fine.Delta) < abs(coarse.Delta),
            })
    pd.DataFrame(rows).to_csv(out / "DIAGNOSTICS_summary.csv", index=False)

    print("\n=== DIAGNOSTICS ===")
    print(pd.DataFrame(rows).to_string(index=False))
    if mi is not None:
        print("\nMilstein:")
        print(mi.round(5).to_string(index=False))
    print("\nOutput:", out.resolve())


if __name__ == "__main__":
    main()