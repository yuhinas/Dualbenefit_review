"""
rdca.py
=======
Reparameterization layer that turns the verified engine (iomodel.py) into a
RESOURCE-DEFENSE (RD) vs COLLECTIVE-ACTION (CA) dual-benefits model.

Core idea
---------
In the engine, *everything* about grouping benefits enters through two
vital-rate vectors:
    p(n) = group productivity (birth rate) at size n,   per-capita = p(n)/n
    m(n) = resident mortality hazard at size n.
The RD <-> CA distinction is therefore entirely a statement about the SHAPE of
p(n)/n and m(n). We build these shapes from mechanistic primitives instead of
asserting Hill/convex forms by hand.

Per-capita productivity:
    w(n) = base * synergy(n) * share(n)
      synergy(n) = 1 + smax * (n-1)^h / (Kben^h + (n-1)^h)   # CA: task synergy / quorum
      share(n)   = 1 / (1 + beta*(n-1))                       # RD: finite shared resource
    p(n) = n * w(n)
At the RD endpoint (smax = 0) the per-capita curve declines monotonically from n = 1;
at the CA endpoint the synergy term makes it rise to an interior optimum before the
sharing term pulls it down (positive density dependence over a range of sizes).

Mortality (CA can also act through survival: vigilance / buffering):
    m(n) = mbase / (1 + sigma * (n-1)^hs / (Ksurv^hs + (n-1)^hs))

Environmental-quality axis theta in [0, 1]:
    theta = 0  -> benign, RD-dominated  (steep sharing, no synergy, flat low mortality)
    theta = 1  -> harsh,  CA-dominated  (weaker sharing, strong synergy, survival augmentation)
All vital-rate primitives interpolate linearly between the RD and CA endpoints.
"""

import numpy as np
import iomodel as M

NMAX = 9


# ---- RD / CA endpoint parameter sets (finalized) ----
# RD endpoint: a TRUE resource-defense ecology. No task synergy (smax = 0), so
#   per-capita performance w(n) declines monotonically from n = 1 (every added
#   member dilutes the finite defended resource). Mortality is flat and low. This
#   yields small, kin-structured groups, the RD prediction.
# CA endpoint: strong task synergy that SATURATES at intermediate n, appreciable
#   sharing, and modest survival augmentation. The productivity optimum and the
#   stationary mode both sit in the interior (mean group size around 6, with
#   negligible probability mass at the n = 9 boundary), so the large-group result
#   is not a truncation artifact of the model's maximum size.
# Both endpoints produce numerically stable cooperative-branch equilibria.
RD_PARAMS = dict(base=1.15, smax=0.0, Kben=2.0, h=2.0,
                 beta=0.40, mbase=0.80, sigma=0.0, Ksurv=3.0, hs=2.0)
CA_PARAMS = dict(base=1.00, smax=1.3, Kben=1.8, h=3.0,
                 beta=0.28, mbase=1.00, sigma=0.40, Ksurv=2.5, hs=2.0)


def _interp(theta, key):
    return (1 - theta) * RD_PARAMS[key] + theta * CA_PARAMS[key]


def vital_rates(theta):
    """Return (m, p) length-9 vectors (n=1..9) for environmental quality theta."""
    base = _interp(theta, 'base')
    smax = _interp(theta, 'smax'); Kben = _interp(theta, 'Kben'); h = _interp(theta, 'h')
    beta = _interp(theta, 'beta')
    mbase = _interp(theta, 'mbase'); sigma = _interp(theta, 'sigma')
    Ksurv = _interp(theta, 'Ksurv'); hs = _interp(theta, 'hs')

    n = np.arange(1, NMAX + 1)
    synergy = 1 + smax * (n - 1) ** h / (Kben ** h + (n - 1) ** h)
    share = 1 / (1 + beta * (n - 1))
    w = base * synergy * share
    p = n * w
    m = mbase / (1 + sigma * (n - 1) ** hs / (Ksurv ** hs + (n - 1) ** hs))
    return m, p


def summarize(R, m, p, k):
    """Compute social-structure summaries from a run() result dict.

    Note on rbar: the mean within-group relatedness is weighted by the stationary
    frequency f(n) of each group size (n >= 2), i.e. it is the expected within-group
    relatedness of a randomly sampled group, not a pair-weighted population average.
    """
    f, r, v = R['f'], R['r'], R['v']
    occ = f[1:10] / f[1:10].sum()                 # distribution over occupied sizes n=1..9
    mode = 1 + int(np.argmax(occ))
    mean = float(sum((nn) * occ[nn - 1] for nn in range(1, 10)))
    wts = f[2:10]
    rbar = float(np.sum(r * wts) / np.sum(wts)) if np.sum(wts) > 0 else float('nan')
    g = M.net_growth(f, m, p, k, R['d'], R['j'])
    sign_changes = int(np.sum(np.diff(np.sign(g)) != 0))
    return dict(occ=occ, mode=mode, mean=mean, rbar=rbar,
                sign_changes=sign_changes, g=g, fd=f[10])


def run_theta(theta, a, mf=1.0, k=7.0, err=0.05, nit=1500):
    m, p = vital_rates(theta)
    d, j = M.defaults()[2], M.defaults()[3]
    R = M.run(m, p, mf, k, err, a, d, j, nit)
    return R, summarize(R, m, p, k)


def sweep(thetas, a, **kw):
    rows = []
    for th in thetas:
        R, s = run_theta(th, a, **kw)
        rows.append((th, s['mode'], s['mean'], s['rbar'], s['sign_changes'], s['fd']))
    return rows


# ============================================================
# BRANCH TRACKING (warm-start continuation)
# ============================================================
# Warm-starting the demographic solve from the previous iteration's equilibrium
# keeps the eco-evolutionary run on a single branch (e.g. the cooperative branch
# when seed='large'), which is the relevant branch for all reported results.

def _seed(name):
    """Initial demographic guess [f0..f9, fd]."""
    if name is None or name == 'wolfram':
        return None                                   # -> iomodel._F_INIT
    if name == 'small':
        return np.array([0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.003, 0.002, 0.05])
    if name == 'large':
        return np.array([0.0, 0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.20, 0.03, 3.0])
    raise ValueError(name)


def run_branch(theta, a, seed='large', mf=1.0, k=7.0, err=0.05, nit=1500):
    """Eco-evolutionary run that warm-starts the demographic solve from the
    previous iteration's equilibrium, so it tracks one branch (e.g. the
    cooperative branch when seed='large')."""
    m, p = vital_rates(theta)
    d, j = M.defaults()[2].copy(), M.defaults()[3].copy()
    state = (m, p, mf, k, err, a, d, j)
    f_init = _seed(seed)
    for _ in range(nit):
        state, f = M.upd(state, f_init=f_init, return_f=True)
        f_init = f                                    # continue on this branch
    m, p, mf, k, err, a, d, j = state
    f, *_ = M._solve_demography(m, p, mf, k, d, j, f_init=f_init)
    r, *_ = M._solve_relatedness(f, m, p, k, d, j, np.zeros(8))
    v, *_ = M._solve_rv(f, m, p, mf, k, d, j)
    R = dict(d=d, j=j, f=f, r=r, v=v)
    return R, summarize(R, m, p, k)


def run_branch_g(theta, a, g=1.0, seed='large', mf=1.0, k=7.0, err=0.05, nit=1500):
    """Like run_branch, but multiplies group productivity p(n) by a fecundity
    multiplier g (so p(n) = g * p0(n)). Used for the fecundity panel: higher g
    lets groups grow through natal recruitment and raises emergent relatedness
    relative to the dilution null."""
    m, p = vital_rates(theta)
    p = g * p
    d, j = M.defaults()[2].copy(), M.defaults()[3].copy()
    state = (m, p, mf, k, err, a, d, j)
    f_init = _seed(seed)
    for _ in range(nit):
        state, f = M.upd(state, f_init=f_init, return_f=True)
        f_init = f
    m, p, mf, k, err, a, d, j = state
    f, *_ = M._solve_demography(m, p, mf, k, d, j, f_init=f_init)
    r, *_ = M._solve_relatedness(f, m, p, k, d, j, np.zeros(8))
    v, *_ = M._solve_rv(f, m, p, mf, k, d, j)
    R = dict(d=d, j=j, f=f, r=r, v=v)
    return R, summarize(R, m, p, k)


# ============================================================
# KIN-STRUCTURE INDICES (natal-recruitment fraction)
# ============================================================
# Reproductive skew is NOT modelled here: within-group fecundity is equal
# (per-capita p(n)/n), so the engine is mechanistically skew-free. The quantities
# the engine DOES derive, onto which the breeding system is mapped as an empirical
# correlate, are:
#   r(n)   within-group relatedness  (already in R['r'])
#   phi(n) natal-recruitment ('family') fraction of group growth at size n:
#            phi(n) = natal / (natal + immigration)
#          phi -> 1 : groups grow as kin families  (associated with singular / high-skew form)
#          phi -> 0 : groups grow by unrelated immigration (plural / low-skew form)
# Skew is thus mapped onto kin structure (r, phi), which is set by membership control
# a, not onto group size: large groups can be high- or low-skew depending on whether
# they are kin (insider control) or non-kin (outsider control).

def kin_structure_indices(R, p, k):
    f, r, d, j = R['f'], R['r'], R['d'], R['j']
    fd = f[10]
    natal = np.array([(1 - d[n - 1]) * p[n - 1] for n in range(1, 9)])   # sizes 1..8, kin route
    immig = np.array([fd * j[n] * k for n in range(1, 9)])               # sizes 1..8, non-kin route
    denom = natal + immig
    phi = np.where(denom > 0, natal / np.where(denom > 0, denom, 1.0), np.nan)
    wn = f[1:9]
    colonize = fd * j[0] * k * f[0]                                      # empty -> size 1 (non-kin)
    tot_natal = float(np.sum(wn * natal))
    tot_immig = float(np.sum(wn * immig) + colonize)
    Phi = tot_natal / (tot_natal + tot_immig) if (tot_natal + tot_immig) > 0 else float('nan')
    rbar = float(np.average(r, weights=f[2:10])) if f[2:10].sum() > 0 else float('nan')
    return dict(phi=phi, Phi=Phi, rbar=rbar)
