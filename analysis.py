"""Post-processing statistico (separato dalla simulazione): Kaplan–Meier, Cox, incertezza e sensitività."""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

from model import CLEARANCE, OFFTARGET, TARGET, transport_summaries


def kaplan_meier(T, event):
    """Curva P_hit(t) = 1 - S(t) con intervallo di confidenza al 95% (Kaplan–Meier, censura a destra)."""
    km = KaplanMeierFitter().fit(T, event)
    ci = km.confidence_interval_survival_function_
    median = km.median_survival_time_ if np.isfinite(km.median_survival_time_) else np.nan   # NaN se non raggiunta
    return pd.DataFrame({"P_hit": 1 - km.survival_function_.iloc[:, 0],
                         "low": 1 - ci.iloc[:, 1], "high": 1 - ci.iloc[:, 0]}), median


def cox_hazard_ratios(df, covariates):
    """Cox su covariate standardizzate (HR per 1 deviazione standard) + test dei rischi proporzionali.
    df: colonne T, event e covariate. Le SE riflettono N traiettorie simulate, non la variabilità tra molecole:
    gli HR sono misure di effetto descrittive, non leggi causali."""
    d = df[["T", "event", *covariates]].copy()
    d[covariates] = (d[covariates] - d[covariates].mean()) / d[covariates].std()
    cph = CoxPHFitter().fit(d, "T", "event")
    ph = proportional_hazard_test(cph, d, time_transform="rank").summary["p"]
    out = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%"]].copy()
    out.columns = ["HR", "low", "high"]
    out["p_PH"] = ph.reindex(out.index).values   # p < 0.05: ipotesi di rischi proporzionali violata
    return out


def propagate(params, cfg, rng, n_samples):
    """Monte Carlo sugli intervalli dei parametri (Param.sample) -> (X, Y): parametri e uscite per campione.
    Uscite da Fokker–Planck: la variabilità di Y è solo incertezza fisica dei parametri, senza errore MC."""
    X = pd.DataFrame([{k: v.sample(rng) for k, v in params.items()} for _ in range(n_samples)])
    Y = pd.DataFrame(transport_summaries(X.to_dict("records"), cfg))
    return X, Y


def sensitivity(X, Y):
    """Correlazione di rango di Spearman tra ogni parametro incerto e ogni uscita (screening globale)."""
    X = X.loc[:, X.nunique() > 1]
    return pd.DataFrame({c: X.corrwith(Y[c], method="spearman") for c in ("P_hit", "T_med")})


def cumulative_incidence(T, outcome, t):
    """P(esito k entro t) per bersaglio, off-target e clearance. Stima empirica esatta: l'unica censura è
    amministrativa (t_max, uguale per tutti). Con rischi concorrenti KM sul solo bersaglio sovrastimerebbe P_hit."""
    t = np.asarray(t)
    return {k: ((outcome == k)[None, :] & (T[None, :] <= t[:, None])).mean(axis=1)
            for k in (TARGET, OFFTARGET, CLEARANCE)}


def pareto_mask(F):
    """True per i punti non dominati. F: (n, m) con tutti gli obiettivi da MINIMIZZARE
    (un punto è dominato se un altro è <= su tutti gli obiettivi e < su almeno uno)."""
    return np.array([not (np.all(F <= f, axis=1) & np.any(F < f, axis=1)).any() for f in F])