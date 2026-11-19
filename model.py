"""Modello fisico: parametri con provenienza, ambiente 1D eterogeneo, Langevin (first passage), Fokker–Planck.

Unità interne coerenti: µm, s (D in µm²/s). Dominio [0, xT]: parete riflettente in x=0 (sangue),
bersaglio assorbente in x=xT. Partenza in x=0.

Convenzione stocastica: isoterma (Hänggi–Klimontovich), cioè equilibrio di Boltzmann anche con D(x) variabile:
    FP:   ∂P/∂t = ∂x[ D (∂x P + P ∂x φ) ],        φ(x) = -ln K(x)   (K = coefficiente di partizione)
    Itô:  dX = D ∂x[ln D - φ] dt + sqrt(2D) dW     (D' è la deriva spuria che rende SDE e FP equivalenti)

Estensioni opzionali, spente se i relativi parametri sono assenti da p (vedi config.yaml: extensions):
    forza esterna uniforme f [kT/µm]: φ -> φ - f x                                   Pe = f L
    binding F⇌B (B immobile): k_bind(x) per compartimento e k_unbind                  Da = k_bind τ_D
    pozzi del primo ordine per compartimento: k_clear (clearance), k_offtarget (captazione off-target)
    2D (Environment2D): pori nel compartimento barriera e orientazione θ con diffusione anisotropa (D_R)

data/molecules.csv, formato lungo (una riga per molecola e parametro):
    molecule,parameter,value,low,high,kind,source,source_id
    parameter ∈ {radius_nm, logP, D_bulk_um2_s}; D_bulk_um2_s (opzionale) sostituisce Stokes–Einstein.
"""
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import trapezoid
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.ndimage import map_coordinates
from scipy.special import exprel

ROOT = Path(__file__).parent
WORKERS = os.cpu_count() or 1      # processi paralleli; 1 = seriale (si imposta da config con configure())
KB = 1.380649e-23
KINDS = {"measured", "calculated", "estimated", "assumed"}
CENSORED, TARGET, OFFTARGET, CLEARANCE = 0, 1, 2, 3      # esiti di una traiettoria
OUTCOMES = {CENSORED: "censored", TARGET: "target", OFFTARGET: "off-target", CLEARANCE: "clearance"}


# ---------------------------------------------------------------- parametri e configurazione
@dataclass
class Param:
    """Parametro con intervallo plausibile e provenienza."""
    value: float
    low: float
    high: float
    kind: str
    source: str

    def __post_init__(self):
        if self.kind not in KINDS or not self.low <= self.value <= self.high:
            raise ValueError(f"parametro non valido: {self}")

    def sample(self, rng):
        """Log-uniforme se strettamente positivo, uniforme altrimenti."""
        if self.low == self.high:
            return self.value
        if self.low > 0:
            return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        return float(rng.uniform(self.low, self.high))

    def grid(self, n):
        """n punti nell'intervallo, con la stessa scala di sample()."""
        return np.geomspace(self.low, self.high, n) if self.low > 0 else np.linspace(self.low, self.high, n)


def load_config(path=ROOT / "config.yaml"):
    return yaml.safe_load(Path(path).read_text())


# ---------------------------------------------------------------- parallelizzazione
def configure(cfg):
    """Imposta il numero di processi da config (workers: null = tutti i core)."""
    global WORKERS
    WORKERS = cfg["workers"] or os.cpu_count() or 1


@lru_cache(maxsize=None)
def _executor(workers):
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")   # niente sovrasottoscrizione BLAS
    return ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))                  # spawn: vale anche su Windows


def pmap(fn, arg_tuples):
    """[fn(*args) per args in arg_tuples] su WORKERS processi; seriale se WORKERS == 1 o un solo compito."""
    arg_tuples = list(arg_tuples)
    if WORKERS == 1 or len(arg_tuples) < 2:
        return [fn(*a) for a in arg_tuples]
    return list(_executor(WORKERS).map(fn, *zip(*arg_tuples), chunksize=max(1, len(arg_tuples) // (4 * WORKERS))))


def env_params(cfg):
    e = cfg["environment"]
    P = {k: Param(**e[k]) for k in ("temperature_K", "viscosity_Pa_s", "partition_slope")}
    for c in e["compartments"]:
        P[f"{c['name']}.width"] = Param(**c["width"])
        P[f"{c['name']}.D_factor"] = Param(**c["D_factor"])
    return P


def reference_params(cfg):
    return {k: Param(**v) for k, v in cfg["reference"].items()}


def extension_params(cfg, groups):
    """Parametri dei meccanismi opzionali richiesti (gruppi di config.yaml: extensions)."""
    return {k: Param(**v) for g in groups for k, v in cfg["extensions"][g].items()}


def load_molecules(path=ROOT / "data" / "molecules.csv"):
    """{molecola: {parametro: Param}}. Parametri obbligatori mancanti -> errore (niente valori inventati)."""
    out = {}
    for name, g in pd.read_csv(path).groupby("molecule", sort=False):
        out[name] = {r.parameter: Param(r.value, r.low, r.high, r.kind, f"{r.source} [{r.source_id}]")
                     for r in g.itertuples()}
        if "logP" not in out[name] or not {"radius_nm", "D_bulk_um2_s"} & out[name].keys():
            raise ValueError(f"{name}: servono logP e (radius_nm o D_bulk_um2_s)")
    return out


def nominal(P):
    return {k: p.value for k, p in P.items()}


def baseline(cfg, homogeneous=False, on=(), **override):
    """Valori nominali di ambiente + molecola di riferimento + gruppi di estensioni attivati (on);
    homogeneous=True elimina contrasti di D e partizione."""
    p = nominal({**env_params(cfg), **reference_params(cfg), **extension_params(cfg, on)})
    if homogeneous:
        p.update({k: 1.0 for k in p if k.endswith("D_factor")}, logP=0.0)
    return {**p, **override}


# ---------------------------------------------------------------- ambiente
class Environment:
    """Profili D(x), φ(x) e tassi cinetici raccordati con tanh (larghezza w), con tabelle per l'integrazione."""

    def __init__(self, p, cfg):
        comps, ig = cfg["environment"]["compartments"], cfg["integration"]
        names = [c["name"] for c in comps]
        self.w, self.xi_min = ig["interface_width_um"], ig["step_min_um"]
        edges = np.cumsum([p[f"{n}.width"] for n in names])
        self.xT, self.interfaces = edges[-1], edges[:-1]
        self.f, self.k_unbind = p.get("force_per_um", 0.0), p.get("k_unbind", 0.0)

        # D_bulk misurato, oppure Stokes–Einstein (solo baseline in fluido semplice)
        self.D_bulk = p.get("D_bulk_um2_s") or (
            KB * p["temperature_K"] / (6 * np.pi * p["viscosity_Pa_s"] * p["radius_nm"] * 1e-9) * 1e12)
        self.lnDc = np.log(self.D_bulk * np.array([p[f"{n}.D_factor"] for n in names]))
        phi = np.array([-p["partition_slope"] * p["logP"] * np.log(10) if c.get("partition") else 0.0
                        for c in comps])
        self.lnD0, self.phi0, self.dlnD, self.dphi = self.lnDc[0], phi[0], np.diff(self.lnDc), np.diff(phi)

        self.xg = np.arange(0, self.xT + ig["step_max_um"], self.w / 5)
        self.D_tab, self.g_tab, _ = self.profiles(self.xg)
        self.D_prime_tab = self.D_prime(self.xg)
        dist = np.abs(self.xg[:, None] - self.interfaces).min(axis=1)
        self.xi_tab = np.clip(ig["step_distance_factor"] * dist, ig["step_min_um"], ig["step_max_um"])
        rates = [[p.get(f"{n}.{k}", 0.0) for n in names] for k in ("k_bind", "k_clear", "k_offtarget")]
        self.rate_tab = np.array([self._smooth(np.array(r), self.xg) for r in rates])   # k_bind, k_clear, k_offtarget
        self.kinetic = bool(self.rate_tab.any())
        self.dtmax_tab = 0.2 / np.maximum(self.rate_tab.sum(axis=0), 1e-9)               # k dt <= 0.2

    def _smooth(self, vals, x):
        """Profilo costante a tratti (un valore per compartimento) raccordato con tanh."""
        return vals[0] + 0.5 * (1 + np.tanh((np.asarray(x)[..., None] - self.interfaces) / self.w)) @ np.diff(vals)

    def profiles(self, x):
        """D(x), g(x) = ∂x(ln D - φ) e φ(x), con l'eventuale forza esterna (φ -> φ - f x)."""
        x = np.asarray(x)
        z = (x[..., None] - self.interfaces) / self.w
        sig, dsig = 0.5 * (1 + np.tanh(z)), 0.5 / np.cosh(z) ** 2 / self.w
        return (np.exp(self.lnD0 + sig @ self.dlnD),
                dsig @ (self.dlnD - self.dphi) + self.f,
                self.phi0 + sig @ self.dphi - self.f * x)

    def D_prime(self, x):
        """Derivata dD/dx del profilo di diffusione."""
        x = np.asarray(x)
        z = (x[..., None] - self.interfaces) / self.w
        dsig = 0.5 / np.cosh(z) ** 2 / self.w
        D = np.exp(self.lnD0 + (0.5 * (1 + np.tanh(z))) @ self.dlnD)
        return D * (dsig @ self.dlnD)

    def rates(self, x):
        """(k_bind, k_clear, k_offtarget) in x [1/s]."""
        return np.array([np.interp(x, self.xg, tab) for tab in self.rate_tab])

    @property
    def tau_D(self):
        """L²/D_serie con D_serie = L/∫dx/D (resistenze in serie): scala di riferimento, non sostituisce T_hit."""
        m = self.xg <= self.xT
        return self.xT * trapezoid(1 / self.D_tab[m], self.xg[m])

    def init_aux(self, n, rng):
        """Variabili di stato oltre a x (nessuna in 1D)."""
        return np.empty((n, 0))

    def _timestep(self, x, D, g, xi, t, t_end):
        """Passo diffusivo xi²/2D, limitato da deriva (|b| dt <= xi) e cinetica (k dt <= 0.2); chiude esattamente a t_end."""
        dt = np.minimum.reduce([xi ** 2 / (2 * D), xi / np.maximum(D * np.abs(g), 1e-12),
                                np.interp(x, self.xg, self.dtmax_tab)])
        last = dt >= t_end - t
        return np.where(last, t_end - t, dt), last

    def step(self, x, aux, t, t_end, rng):
        D, g, xi, Dp = (np.interp(x, self.xg, tab) for tab in (self.D_tab, self.g_tab, self.xi_tab, self.D_prime_tab))
        dt, last = self._timestep(x, D, g, xi, t, t_end)
        N = rng.standard_normal(x.size)
        # Milstein: drift + diffusione + correzione
        x_new = x + D * g * dt + np.sqrt(2 * D * dt) * N + 0.5 * Dp * dt * (N**2 - 1)
        x_new = np.abs(x_new)  # riflessione in x=0
        return x_new, aux, np.where(last, t_end, t + dt), D * dt


class Environment2D(Environment):
    """Estensione 2D: periodica in y (periodo W), pori nel compartimento barriera (porosity = frazione di W)
    e orientazione θ con diffusione rotazionale D_R: D_xx = D(1 + a cos2θ), D_yy = D(1 - a cos2θ), D_xy = D a sin2θ,
    con a = (D∥ - D⊥)/(D∥ + D⊥). Nel poro il compartimento ha il D del compartimento a monte (barriera assente)."""

    def __init__(self, p, cfg):
        super().__init__(p, cfg)
        names = [c["name"] for c in cfg["environment"]["compartments"]]
        k, w = names.index(cfg["geometry_2d"]["pore_compartment"]), self.w
        por, self.W = p.get("porosity", 0.0), p.get("pore_period_um", 1.0)
        self.a, self.D_R, self.dx, self.dy = p.get("anisotropy", 0.0), p.get("D_R_per_s", 0.0), w / 5, w / 5
        yg = np.arange(-self.W / 2, self.W / 2 + self.dy / 2, self.dy)
        X, Y = np.meshgrid(self.xg, yg, indexing="ij")
        # indicatore regolare del compartimento barriera (E) e della sua apertura (M) con le derivate
        sig = lambda z: 0.5 * (1 + np.tanh(z))
        dsig = lambda z: 0.5 / np.cosh(z) ** 2 / w
        za, zb = (X - self.interfaces[k - 1]) / w, (X - self.interfaces[k]) / w
        E, dE = sig(za) - sig(zb), dsig(za) - dsig(zb)
        zy = (por * self.W / 2 - np.abs(Y)) / w
        inside = 0 < por < 1
        M = np.where(inside, sig(zy), float(por >= 1))
        dM = np.where(inside, -np.sign(Y) * dsig(zy), 0.0)
        delta = self.lnDc[k - 1] - self.lnDc[k]
        xi = np.where(inside & (E > 1e-3), self.xi_min, self.xi_tab[:, None])
        self.tabs = [np.exp(np.log(self.D_tab)[:, None] + delta * E * M), self.g_tab[:, None] + delta * dE * M,
                     delta * E * dM, xi]                                            # D, gx, gy, passo

    def init_aux(self, n, rng):
        return np.column_stack([rng.uniform(-self.W / 2, self.W / 2, n), rng.uniform(0, 2 * np.pi, n)])

    def step(self, x, aux, t, t_end, rng):
        y, th = aux.T
        D, gx, gy, xi = (map_coordinates(tab, [x / self.dx, (y + self.W / 2) / self.dy], order=1, mode="nearest")
                         for tab in self.tabs)
        dt, last = self._timestep(x, D, np.hypot(gx, gy), xi, t, t_end)
        a = self.a * exprel(-4 * self.D_R * dt)                  # anisotropia efficace: media di cos2θ sul passo
        c2, s2 = np.cos(2 * th), np.sin(2 * th)
        n_, m_ = np.stack([np.cos(th), np.sin(th)], 1), np.stack([-np.sin(th), np.cos(th)], 1)
        z = rng.standard_normal((x.size, 3))
        noise = np.sqrt(2 * D * dt)[:, None] * (np.sqrt(1 + a)[:, None] * z[:, :1] * n_
                                                + np.sqrt(1 - a)[:, None] * z[:, 1:2] * m_)
        drift = (D * dt)[:, None] * np.stack([(1 + a * c2) * gx + a * s2 * gy, a * s2 * gx + (1 - a * c2) * gy], 1)
        dr = drift + noise
        aux = np.column_stack([(y + dr[:, 1] + self.W / 2) % self.W - self.W / 2, th + np.sqrt(2 * self.D_R * dt) * z[:, 2]])
        return np.abs(x + dr[:, 0]), aux, np.where(last, t_end, t + dt), D * dt * (1 + a * c2)


def make_environment(p, cfg):
    """Environment2D se p contiene parametri 2D/anisotropi, altrimenti Environment 1D."""
    return (Environment2D if {"porosity", "anisotropy"} & p.keys() else Environment)(p, cfg)


# ---------------------------------------------------------------- Langevin: primo passaggio
def first_passage(env, n, t_max, rng, x0=0.0):
    """n traiettorie fino al primo esito o a t_max. Ritorna (T, outcome, x_end): tempo osservato, esito
    (CENSORED, TARGET, OFFTARGET, CLEARANCE) e posizione finale delle traiettorie censurate (NaN altrimenti).
    Binding: all'evento di legame la particella resta ferma per un tempo Exp(k_unbind) (stato B immobile)."""
    T, outcome, x_end = np.full(n, t_max), np.zeros(n, np.int8), np.full(n, np.nan)
    idx, x, t, aux = np.arange(n), np.full(n, float(x0)), np.zeros(n), env.init_aux(n, rng)
    while idx.size:
        x_old, t_old = x, t
        x, aux, t, Ddt = env.step(x, aux, t, t_max, rng)
        # attraversamento tra due passi: probabilità del ponte browniano (D costante vicino al bersaglio)
        p_cross = np.exp(-np.maximum((env.xT - x_old) * (env.xT - x), 0) / Ddt)
        fate = np.where((x >= env.xT) | (rng.random(x.size) < p_cross), TARGET, CENSORED)
        if env.kinetic:
            dt, (kon, kc, ko) = t - t_old, env.rates(x_old)
            lost = (fate == CENSORED) & (rng.random(x.size) < -np.expm1(-(kc + ko) * dt))
            fate[lost] = np.where(rng.random(lost.sum()) * (kc + ko)[lost] < ko[lost], OFFTARGET, CLEARANCE)
            bind = (fate == CENSORED) & (rng.random(x.size) < -np.expm1(-kon * dt))
            t = np.where(bind, np.minimum(t + rng.exponential(1 / max(env.k_unbind, 1e-12), x.size), t_max), t)
        done = (fate != CENSORED) | (t >= t_max)
        T[idx[done]], outcome[idx[done]] = t[done], fate[done]
        alive = done & (fate == CENSORED)
        x_end[idx[alive]] = x[alive]
        idx, x, t, aux = idx[~done], x[~done], t[~done], aux[~done]
    return T, outcome, x_end


def first_passage_parallel(env, n, t_max, rng, x0=0.0):
    """first_passage con n traiettorie divise in WORKERS blocchi, ciascuno con un generatore indipendente (rng.spawn).
    Riproducibile a parità di seed e di WORKERS (il numero di processi cambia i flussi casuali, non la statistica)."""
    k = min(WORKERS, n)
    parts = pmap(first_passage, [(env, len(c), t_max, g, x0) for c, g in zip(np.array_split(np.arange(n), k), rng.spawn(k))])
    return tuple(np.concatenate(p) for p in zip(*parts))


def simulate(env, rng, t_max, n=None, se=None, batch=2000, max_n=200_000):
    """n traiettorie, oppure batch successivi finché SE(P_hit) <= se.
    Ritorna (T, outcome, history) con history = [(N, P_hit, SE)] come diagnostica di convergenza."""
    T, outcome, history = np.empty(0), np.empty(0, np.int8), []
    while True:
        t_, o_, _ = first_passage_parallel(env, batch if n is None else min(batch, n - T.size), t_max, rng)
        T, outcome = np.append(T, t_), np.append(outcome, o_)
        p = (outcome == TARGET).mean()
        history.append((outcome.size, p, np.sqrt(p * (1 - p) / outcome.size)))
        if (n is not None and T.size >= n) or (se is not None and 0 < p and history[-1][2] <= se) \
                or T.size >= max_n:
            return T, outcome, history


# ---------------------------------------------------------------- Fokker–Planck
def _bernoulli(z):
    out = np.ones_like(z)
    m = z != 0
    out[m] = z[m] / np.expm1(z[m])
    return out


class FokkerPlanck:
    """Volumi finiti con flusso di Scharfetter–Gummel (esatto per φ lineare nella cella, preserva Boltzmann).
    Il generatore è simmetrizzabile (anche con binding, per bilancio dettagliato), quindi la soluzione è
    spettrale ed esatta nel tempo. Pozzi: bersaglio (assorbimento in xT), off-target e clearance (tassi)."""

    def __init__(self, env, cells_per_width=4):
        n = int(np.ceil(env.xT * cells_per_width / env.w))
        self.h = h = env.xT / n
        self.x = (np.arange(n) + 0.5) * h
        Df = env.profiles(np.arange(1, n) * h)[0]
        phi = env.profiles(self.x)[2]
        dphi = np.diff(phi)
        right, left = Df / h**2 * _bernoulli(dphi), Df / h**2 * _bernoulli(-dphi)   # tassi j->j+1 e j+1->j
        kon, kc, ko = env.rates(self.x)
        dT = 2 * env.profiles(np.array([env.xT]))[0][0] / h**2                       # assorbimento in xT
        diag = -(kon + kc + ko)
        diag[:-1] -= right
        diag[1:] -= left
        diag[-1] -= dT
        s = dphi / 2
        off = Df / h**2 * np.where(s == 0, 1.0, s / np.sinh(np.where(s == 0, 1.0, s)))
        sq = np.exp(-(phi - phi[0]) / 2)                       # π^{1/2}, con π ∝ exp(-φ)
        sinks = {TARGET: np.r_[np.zeros(n - 1), dT], OFFTARGET: ko, CLEARANCE: kc}
        if kon.any():       # stato legato immobile: blocco aggiuntivo (π_B = π_F k_bind/k_unbind), matrice densa 2n x 2n
            kb = env.k_unbind
            C = np.block([[np.diag(diag) + np.diag(off, 1) + np.diag(off, -1), np.diag(np.sqrt(kon * kb))],
                          [np.diag(np.sqrt(kon * kb)), -kb * np.eye(n)]])
            self.lam, self.V = eigh(C)
            sq, sinks = np.r_[sq, sq * np.sqrt(kon / kb)], {k: np.r_[v, np.zeros(n)] for k, v in sinks.items()}
        else:
            self.lam, self.V = eigh_tridiagonal(diag, off)
        self.sq, self.sinks, self.n = sq, sinks, n
        self.mode_w = (sq @ self.V) * self.V[0]                # partenza libera nella cella 0

    def survival(self, t):
        """Probabilità di non aver ancora raggiunto alcun esito."""
        return np.exp(np.outer(np.atleast_1d(t), self.lam)) @ self.mode_w

    def outcomes(self, t):
        """Probabilità cumulate {TARGET, OFFTARGET, CLEARANCE} entro t (integrale del flusso nei pozzi)."""
        gain = np.expm1(np.outer(np.atleast_1d(t), self.lam)) / self.lam
        return {k: gain @ (((v * self.sq) @ self.V) * self.V[0]) for k, v in self.sinks.items()}

    def hit_density(self, t):
        """Densità del tempo di arrivo al bersaglio, dP_target/dt."""
        return np.exp(np.outer(np.atleast_1d(t), self.lam)) @ (((self.sinks[TARGET] * self.sq) @ self.V) * self.V[0])

    def density(self, t):
        """P(x, t) delle particelle non ancora uscite (libere + legate), nei centri cella."""
        return (self.sq * (self.V @ (np.exp(self.lam * t) * self.V[0])) / self.h).reshape(-1, self.n).sum(axis=0)


def transport_summary(p, cfg):
    """Esiti a t_max, tempi di arrivo e numeri adimensionali da Fokker–Planck (deterministico, senza errore MC).
    T_med: mediana di T_hit (NaN se P_hit < 0.5); T_cond: mediana dei tempi di arrivo dei soli arrivi riusciti."""
    env = Environment(p, cfg)
    t = np.geomspace(1e-4 * cfg["t_max_s"], cfg["t_max_s"], 400)
    out = FokkerPlanck(env, cfg["integration"]["fp_cells_per_width"]).outcomes(t)
    P = out[TARGET]
    return {"P_hit": P[-1], "P_offtarget": out[OFFTARGET][-1], "P_clear": out[CLEARANCE][-1],
            "T_med": np.interp(0.5, P, t) if P[-1] >= 0.5 else np.nan,
            "T_cond": np.interp(P[-1] / 2, P, t) if P[-1] > 0 else np.nan,
            "tau_D": env.tau_D, "Pe": env.f * env.xT, "Da": env.rate_tab[0].max() * env.tau_D}


def transport_summaries(ps, cfg):
    """transport_summary per una lista di parametri (sweep, incertezza, candidati), in parallelo."""
    return pmap(transport_summary, [(p, cfg) for p in ps])