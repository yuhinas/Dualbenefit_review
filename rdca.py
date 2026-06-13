"""
rdca.py
=======
Reparameterization layer that turns the verified v15 engine (iomodel.py) into a
RESOURCE-DEFENSE (RD) vs COLLECTIVE-ACTION (CA) dual-benefits model.

Core idea
---------
In the v15 engine, *everything* about grouping benefits enters through two
vital-rate vectors:
    p(n) = group productivity (birth rate) at size n,   per-capita = p(n)/n
    m(n) = resident mortality hazard at size n.
The RD <-> CA distinction is therefore entirely a statement about the SHAPE of
p(n)/n and m(n).  We BUILD these shapes from mechanistic primitives instead of
asserting Hill/convex forms by hand, which is what answers Reviewer #3's
"arbitrary functional form" objection.

Per-capita productivity:
    w(n) = base * synergy(n) * share(n)
      synergy(n) = 1 + smax * (n-1)^h / (Kben^h + (n-1)^h)   # CA: task synergy / quorum
      share(n)   = 1 / (1 + beta*(n-1))                       # RD: finite shared resource
    p(n) = n * w(n)
This product is automatically hump-shaped (rise from synergy, fall from sharing),
as the v15 framework requires.

Mortality (CA can also act through survival: vigilance / buffering):
    m(n) = mbase / (1 + sigma * (n-1)^hs / (Ksurv^hs + (n-1)^hs))

Environmental-quality axis theta in [0,1]:
    theta = 0  -> benign, RD-dominated  (steep sharing, weak synergy, flat survival, low mortality)
    theta = 1  -> harsh,  CA-dominated  (weak sharing, strong synergy, group-buffered survival, high mortality)
Parameters interpolate linearly between the RD and CA endpoints.
"""

import numpy as np
import iomodel as M

NMAX = 9


# ---- RD / CA endpoint parameter sets (edit these to explore) ----
# RD: small early hump (defensible-resource benefit) then steep sharing; flat, low mortality.
# CA: strong synergy peaking at intermediate n, weak sharing, group-buffered survival; harsh.
RD_PARAMS = dict(base=1.05, smax=0.9, Kben=1.2, h=2.0,
                 beta=0.35, mbase=0.85, sigma=0.0, Ksurv=3.0, hs=2.0)
CA_PARAMS = dict(base=1.10, smax=1.8, Kben=2.5, h=3.0,
                 beta=0.26, mbase=1.00, sigma=1.2, Ksurv=3.0, hs=2.0)


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
    """Compute social-structure summaries from a run() result dict."""
    f, r, v = R['f'], R['r'], R['v']
    occ = f[1:10] / f[1:10].sum()                 # distribution over occupied sizes n=1..9
    mode = 1 + int(np.argmax(occ))
    mean = float(sum((nn) * occ[nn - 1] for nn in range(1, 10)))
    # mean within-group relatedness, weighted by how many groups of each size (n>=2)
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
# BRANCH TRACKING (warm-start continuation) + BISTABILITY PROBE
# ============================================================
# Under strong CA the demographic system is bistable (an empty/collapse root
# f0=1, fd=0, and a large-group cooperative root).  The Wolfram default start
# lands on collapse.  These tools track the cooperative branch and detect the
# multiplicity explicitly.

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


def bistability_probe(m, p, mf, k, d, j, seeds=('small', 'large', 'wolfram'), tol=1e-8):
    """Multi-start the demographic solver at FIXED strategies and return the
    distinct valid equilibria found (deduplicated)."""
    found = []
    for name in seeds:
        x, info, ier, msg = M._solve_demography(m, p, mf, k, d, j, f_init=_seed(name))
        res = np.max(np.abs(M._demographic_residuals(x, m, p, mf, k, d, j)))
        valid = (ier == 1 and res < tol and (x[:10] >= -1e-9).all()
                 and abs(x[:10].sum() - 1) < 1e-7 and x[10] >= -1e-9)
        if not valid:
            continue
        occ = x[1:10] / max(x[1:10].sum(), 1e-12)
        mean = float(sum(n * occ[n - 1] for n in range(1, 10)))
        eq = dict(seed=name, mode=1 + int(np.argmax(occ)), mean=mean,
                  fd=float(x[10]), f0=float(x[0]), f=x.copy())
        if not any(abs(e['mean'] - eq['mean']) < 1e-4 and abs(e['fd'] - eq['fd']) < 1e-4
                   for e in found):
            found.append(eq)
    return found


# ============================================================
# INTERPRETIVE-OVERLAY INDICES FOR THE BREEDING SYSTEM / SKEW
# ============================================================
# Reproductive skew is NOT modelled here: within-group fecundity is equal
# (per-capita p(n)/n), i.e. the engine is mechanistically skew-free.  These are
# the quantities the engine DOES derive, onto which the breeding system / skew
# is mapped as an empirical correlate (see EXPERIMENT_DESIGN.md sec 7):
#   r(n)   within-group relatedness  (already in R['r'])
#   phi(n) natal-recruitment ('family') fraction of group growth at size n:
#            phi(n) = natal / (natal + immigration)
#          phi -> 1 : groups grow as kin families  (-> singular / high-skew form)
#          phi -> 0 : groups grow by unrelated immigration (-> plural / low-skew form)
# The conceptual payoff: skew is mapped onto kin structure (r, phi), which is set
# by membership control a, NOT onto group size.  Large groups can be high- or
# low-skew depending on whether they are kin (insider control) or non-kin
# (outsider control).

def kin_structure_indices(R, p, k):
    f, r, d, j = R['f'], R['r'], R['d'], R['j']
    fd = f[10]
    natal = np.array([(1 - d[n - 1]) * p[n - 1] for n in range(1, 9)])   # sizes 1..8, kin route
    immig = np.array([fd * j[n] * k for n in range(1, 9)])               # sizes 1..8, non-kin route
    denom = natal + immig
    phi = np.where(denom > 0, natal / np.where(denom > 0, denom, 1.0), np.nan)
    # population-weighted family fraction of ALL membership additions
    wn = f[1:9]
    colonize = fd * j[0] * k * f[0]                                      # empty -> size 1 (non-kin)
    tot_natal = float(np.sum(wn * natal))
    tot_immig = float(np.sum(wn * immig) + colonize)
    Phi = tot_natal / (tot_natal + tot_immig) if (tot_natal + tot_immig) > 0 else float('nan')
    rbar = float(np.average(r, weights=f[2:10])) if f[2:10].sum() > 0 else float('nan')
    return dict(phi=phi, Phi=Phi, rbar=rbar)


# ============================================================
# LITERAL BOX-3 CORRESPONDENCE: additive B_A(n) - C(n) vital rates
# ============================================================
# Box 3 writes the per-capita increment additively as psi(n) = B_A(n) - C(n)
# with a Hill/Allee CA benefit and a convex RD cost. This builder puts the
# SAME B_A and C into the v15 engine, so the dual-benefits paper can feed its
# own Box-3 parameters straight through the derived model:
#   B_A(n) = B_max (n-1)^h / (K^h + (n-1)^h)          # CA benefit (Hill / Allee)
#   C(n)   = C_up n^eta                                # RD crowding/conflict cost
#   psi(n) = B_A(n) - C(n)                             # per-capita increment (Box 3)
#   w(n)   = w1 + sum_{k=1}^{n-1} psi(k)               # per-capita performance
#   p(n)   = n * w(n)        m(n) = m_const            # v15 inputs
# The CA/RD axis (sign of psi) is then IDENTICAL to Box 3; everything downstream
# (relatedness, consensus, bistability) is derived by the engine instead of
# assumed.

# Box 3 canonical parameters (Section S4 of the supplement)
BOX3_PARAMS = dict(B_max=0.60, K=4.5, h=4.5, C_up=0.0055, eta=2.2)


def vital_rates_box3(B_max=0.60, K=4.5, h=4.5, C_up=0.0055, eta=2.2,
                     w1=1.0, m_const=1.0, nmax=NMAX):
    n = np.arange(1, nmax + 1, dtype=float)
    BA = B_max * (n - 1.0) ** h / (K ** h + (n - 1.0) ** h)
    C = C_up * n ** eta
    psi = BA - C                                  # psi[i] = psi(n=i+1)
    w = np.empty(nmax)
    w[0] = w1                                     # w(n=1)
    for i in range(1, nmax):
        w[i] = w[i - 1] + psi[i - 1]              # w(n) = w(n-1) + psi(n-1)
    p = n * w
    m = np.full(nmax, float(m_const))
    return m, p, dict(BA=BA, C=C, psi=psi, w=w)


def box3_assumed_relatedness(r0=0.30, kappa=0.20, nmax=NMAX):
    """Box 3's ASSUMED kin-dilution curve r(n) = r0/(1+kappa(n-1)), for n=2..nmax,
    to overlay against the v15 EMERGENT r(n)."""
    n = np.arange(2, nmax + 1, dtype=float)
    return r0 / (1.0 + kappa * (n - 1.0))
