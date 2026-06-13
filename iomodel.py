"""
iomodel.py
==========
Faithful Python port of the Wolfram Mathematica model in
`Appendix_S1_GroupSizeModel.wl` (Shen, Liu & Rubenstein, "The evolution of
group size and kin structure in complex societies").

Design priority: CORRECTNESS / FIDELITY, not speed.

Every equilibrium system (demographic f, relatedness r, reproductive value v)
and every best-response expression is transcribed VERBATIM from the Wolfram
source, so that a line-by-line visual diff against the .wl file is possible.
We deliberately do NOT algebraically simplify any expression.

Conventions (matching the .wl file exactly):
  nmax = 9
  m = [m1..m9]          resident mortality hazards            (length 9, n=1..9)
  p = [p1..p9]          group productivity (birth rate)        (length 9, n=1..9)
  d = [d1..d9]          dispersal probabilities, d9 == 1       (length 9, n=1..9)
  j = [j0..j9]          joining probabilities,    j9 == 0      (length 10, n=0..9)
  mf                    floater mortality hazard
  k                     floater-group encounter rate
  err                   selection strength (smaller => sharper best response)
  a                     out-group control (0 = insider control, 1 = outsider control)

State vector for run/upd (matching the .wl `upd` argument):
  state = (m, p, mf, k, err, a, d, j)
"""

import numpy as np
from scipy.optimize import fsolve

NMAX = 9


# ============================================================
# EQUILIBRIUM RESIDUALS  (each block transcribed verbatim)
# ============================================================

def _demographic_residuals(x, m, p, mf, k, d, j):
    """Residuals of the f0..f9, fd system (Wolfram `feq`)."""
    f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, fd = x
    m1, m2, m3, m4, m5, m6, m7, m8, m9 = m
    p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    d1, d2, d3, d4, d5, d6, d7, d8, d9 = d
    j0, j1, j2, j3, j4, j5, j6, j7, j8, j9 = j
    return [
        -f0*fd*j0*k + f1*m1,
        f0*fd*j0*k - f1*fd*j1*k - f1*m1 + 2*f2*m2 - (1 - d1)*f1*p1,
        f1*fd*j1*k - f2*fd*j2*k - 2*f2*m2 + 3*f3*m3 + (1 - d1)*f1*p1 - (1 - d2)*f2*p2,
        f2*fd*j2*k - f3*fd*j3*k - 3*f3*m3 + 4*f4*m4 + (1 - d2)*f2*p2 - (1 - d3)*f3*p3,
        f3*fd*j3*k - f4*fd*j4*k - 4*f4*m4 + 5*f5*m5 + (1 - d3)*f3*p3 - (1 - d4)*f4*p4,
        f4*fd*j4*k - f5*fd*j5*k - 5*f5*m5 + 6*f6*m6 + (1 - d4)*f4*p4 - (1 - d5)*f5*p5,
        f5*fd*j5*k - f6*fd*j6*k - 6*f6*m6 + 7*f7*m7 + (1 - d5)*f5*p5 - (1 - d6)*f6*p6,
        f6*fd*j6*k - f7*fd*j7*k - 7*f7*m7 + 8*f8*m8 + (1 - d6)*f6*p6 - (1 - d7)*f7*p7,
        f7*fd*j7*k - f8*fd*j8*k - 8*f8*m8 + 9*f9*m9 + (1 - d7)*f7*p7 - (1 - d8)*f8*p8,
        f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 - 1,
        (-fd*(f0*j0 + f1*j1 + f2*j2 + f3*j3 + f4*j4 + f5*j5 + f6*j6 + f7*j7 + f8*j8)*k
         - fd*mf + d1*f1*p1 + d2*f2*p2 + d3*f3*p3 + d4*f4*p4 + d5*f5*p5 + d6*f6*p6
         + d7*f7*p7 + d8*f8*p8 + f9*p9),
    ]


def _relatedness_residuals(r, f, m, p, k, d, j):
    """Residuals of the r2..r9 system (Wolfram `req`).  Each is r_n - RHS."""
    r2, r3, r4, r5, r6, r7, r8, r9 = r
    f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, fd = f
    m1, m2, m3, m4, m5, m6, m7, m8, m9 = m
    p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    d1, d2, d3, d4, d5, d6, d7, d8, d9 = d
    j0, j1, j2, j3, j4, j5, j6, j7, j8, j9 = j
    return [
        r2 - (f1*(p1 - d1*p1) + 3*f3*m3*r3) / (3*f3*m3 + f1*(fd*j1*k + p1 - d1*p1)),
        r3 - (f2*(fd*j2*k*r2 - (-1 + d2)*p2*(1 + 2*r2)) + 12*f4*m4*r4) / (3*(4*f4*m4 + f2*(fd*j2*k + p2 - d2*p2))),
        r4 - (f3*(3*fd*j3*k*r3 - (-1 + d3)*p3*(1 + 5*r3)) + 30*f5*m5*r5) / (6*(5*f5*m5 + f3*(fd*j3*k + p3 - d3*p3))),
        r5 - (f4*(6*fd*j4*k*r4 - (-1 + d4)*p4*(1 + 9*r4)) + 60*f6*m6*r6) / (10*(6*f6*m6 + f4*(fd*j4*k + p4 - d4*p4))),
        r6 - (f5*(10*fd*j5*k*r5 - (-1 + d5)*p5*(1 + 14*r5)) + 105*f7*m7*r7) / (15*(7*f7*m7 + f5*(fd*j5*k + p5 - d5*p5))),
        r7 - (f6*(15*fd*j6*k*r6 - (-1 + d6)*p6*(1 + 20*r6)) + 168*f8*m8*r8) / (21*(8*f8*m8 + f6*(fd*j6*k + p6 - d6*p6))),
        r8 - (f7*(21*fd*j7*k*r7 - (-1 + d7)*p7*(1 + 27*r7)) + 252*f9*m9*r9) / (28*(9*f9*m9 + f7*(fd*j7*k + p7 - d7*p7))),
        r9 - (-((-28*fd*j8*k*r8 + (-1 + d8)*p8*(1 + 35*r8)) / (36*(fd*j8*k + p8 - d8*p8)))),
    ]


def _rv_residuals(v, f, m, p, mf, k, d, j):
    """Residuals of the v1..v9, vf system (Wolfram `veq`).  Each is `expr` (== 0)."""
    v1, v2, v3, v4, v5, v6, v7, v8, v9, vf = v
    f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, fd = f
    m1, m2, m3, m4, m5, m6, m7, m8, m9 = m
    p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    d1, d2, d3, d4, d5, d6, d7, d8, d9 = d
    j0, j1, j2, j3, j4, j5, j6, j7, j8, j9 = j
    return [
        -m1*v1 + fd*j1*k*(-v1 + v2) + (1 - d1)*p1*(-v1 + 2*v2) + d1*p1*vf,
        m2*(v1 - v2) - m2*v2 + fd*j2*k*(-v2 + v3) + (1 - d2)*p2*(-v2 + (3*v3)/2) + (d2*p2*vf)/2,
        2*m3*(v2 - v3) - m3*v3 + fd*j3*k*(-v3 + v4) + (1 - d3)*p3*(-v3 + (4*v4)/3) + (d3*p3*vf)/3,
        3*m4*(v3 - v4) - m4*v4 + fd*j4*k*(-v4 + v5) + (1 - d4)*p4*(-v4 + (5*v5)/4) + (d4*p4*vf)/4,
        4*m5*(v4 - v5) - m5*v5 + fd*j5*k*(-v5 + v6) + (1 - d5)*p5*(-v5 + (6*v6)/5) + (d5*p5*vf)/5,
        5*m6*(v5 - v6) - m6*v6 + fd*j6*k*(-v6 + v7) + (1 - d6)*p6*(-v6 + (7*v7)/6) + (d6*p6*vf)/6,
        6*m7*(v6 - v7) - m7*v7 + fd*j7*k*(-v7 + v8) + (1 - d7)*p7*(-v7 + (8*v8)/7) + (d7*p7*vf)/7,
        7*m8*(v7 - v8) - m8*v8 + fd*j8*k*(-v8 + v9) + (1 - d8)*p8*(-v8 + (9*v9)/8) + (d8*p8*vf)/8,
        ((f1*v1 + 2*f2*v2 + 3*f3*v3 + 4*f4*v4 + 5*f5*v5 + 6*f6*v6 + 7*f7*v7 + 8*f8*v8 + 9*f9*v9 + fd*vf)
         / (f1 + 2*f2 + 3*f3 + 4*f4 + 5*f5 + 6*f6 + 7*f7 + 8*f8 + 9*f9 + fd) - 1),
        (f0*j0*k*(v1 - vf) + f1*j1*k*(v2 - vf) + f2*j2*k*(v3 - vf) + f3*j3*k*(v4 - vf)
         + f4*j4*k*(v5 - vf) + f5*j5*k*(v6 - vf) + f6*j6*k*(v7 - vf) + f7*j7*k*(v8 - vf)
         + f8*j8*k*(v9 - vf) - mf*vf),
    ]


# ============================================================
# BEST RESPONSE  (Wolfram `upd` Step 4, transcribed verbatim)
# ============================================================

def _best_response(v, r, a, err):
    """Return (d_br[1..8], j_br[0..8]) using the verbatim Tanh expressions."""
    v1, v2, v3, v4, v5, v6, v7, v8, v9, vf = v
    r2, r3, r4, r5, r6, r7, r8, r9 = r

    def S(x):
        return (1 + np.tanh(x / err)) / 2

    d_br = [
        S(v1 - 2*v2 + vf),
        S(-((-2 + a)*(1 + r2)*v2) + (-3 + a - 3*r2 + 2*a*r2)*v3 + (1 + r2 - a*r2)*vf),
        S(-((-1 + a)*(1 + 2*r3)*(3*v3 - 4*v4 + vf)) + a*(v3 + 2*r3*v3 - 2*(1 + r3)*v4 + vf)),
        S(-((-1 + a)*(1 + 3*r4)*(4*v4 - 5*v5 + vf)) + a*(v4 + 3*r4*v4 - (2 + 3*r4)*v5 + vf)),
        S(-((-1 + a)*(1 + 4*r5)*(5*v5 - 6*v6 + vf)) + a*(v5 + 4*r5*v5 - 2*(1 + 2*r5)*v6 + vf)),
        S(-((-1 + a)*(1 + 5*r6)*(6*v6 - 7*v7 + vf)) + a*(v6 + 5*r6*v6 - (2 + 5*r6)*v7 + vf)),
        S(-((-1 + a)*(1 + 6*r7)*(7*v7 - 8*v8 + vf)) + a*(v7 + 6*r7*v7 - 2*(1 + 3*r7)*v8 + vf)),
        S(-((-1 + a)*(1 + 7*r8)*(8*v8 - 9*v9 + vf)) + a*(v8 + 7*r8*v8 - (2 + 7*r8)*v9 + vf)),
    ]
    j_br = [
        S(v1 - vf),
        S((-1 + a)*v1 + v2 - a*vf),
        S(2*(-1 + a)*(1 + r2)*(v2 - v3) + a*(v3 - vf)),
        S(3*(-1 + a)*(1 + 2*r3)*(v3 - v4) + a*(v4 - vf)),
        S(4*(-1 + a)*(1 + 3*r4)*(v4 - v5) + a*(v5 - vf)),
        S(5*(-1 + a)*(1 + 4*r5)*(v5 - v6) + a*(v6 - vf)),
        S(6*(-1 + a)*(1 + 5*r6)*(v6 - v7) + a*(v7 - vf)),
        S(7*(-1 + a)*(1 + 6*r7)*(v7 - v8) + a*(v8 - vf)),
        S(8*(-1 + a)*(1 + 7*r8)*(v8 - v9) + a*(v9 - vf)),
    ]
    return d_br, j_br


def _best_response_closed(v, r, a, err):
    """Closed-form best response, transparent to the inclusive-fitness
    derivation in the appendix. Returns (d_br[1..8], j_br[0..8]).

    Verified equal to the verbatim _best_response to machine precision over
    random inputs (see verify_core.py). The key term, absent from a naive
    reduction, is r_out*(v(n+1) - vF) in the insider effect: the focal insider
    is related to the joiner by r_out, and the joiner's value changes from vF
    (floater, the alternative to joining) to v(n+1) (new member).

        r(n)   : r(1)=1, r=[r2..r9]              R(n) = 1 + (n-1) r(n)   (= r_out * n for a natal offspring)
        DI_out(n) = (1 + r_out n) v(n+1) - vF - r_out n v(n)
        DI_in(n)  = [1 + (n-1) r(n)] (v(n+1) - v(n)) + r_out (v(n+1) - vF)
        Delta(n)  = a DI_out(n) + (1-a) n DI_in(n)
        j_BR(n) = Pjoin(Delta(n)) with r_out = 0          (foreign floater)
        d_BR(n) = 1 - Pjoin(Delta(n)) with r_out = R(n)/n (natal offspring)
        j_BR(0) = Pjoin(v(1) - vF)                        (empty territory, a = 1)
    """
    vf = v[9]
    V = lambda n: v[n - 1]
    rr = lambda n: 1.0 if n == 1 else r[n - 2]
    R = lambda n: 1.0 + (n - 1) * rr(n)
    S = lambda x: (1 + np.tanh(x / err)) / 2
    dIout = lambda n, ro: (1 + ro * n) * V(n + 1) - vf - ro * n * V(n)
    dIin = lambda n, ro: R(n) * (V(n + 1) - V(n)) + ro * (V(n + 1) - vf)
    Delta = lambda n, ro: a * dIout(n, ro) + (1 - a) * n * dIin(n, ro)
    d_br = [1 - S(Delta(n, R(n) / n)) for n in range(1, 9)]            # d1..d8 (natal)
    j_br = [S(V(1) - vf)] + [S(Delta(n, 0.0)) for n in range(1, 9)]    # j0, then j1..j8 (floater)
    return d_br, j_br


# ============================================================
# SOLVERS
# ============================================================

# Initial guesses matching the Wolfram FindRoot starting points.
_F_INIT = np.array([0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 1.0])
_V_INIT = np.ones(10)


def _solve_demography(m, p, mf, k, d, j, f_init=None):
    start = _F_INIT if f_init is None else np.asarray(f_init, float)
    sol = fsolve(_demographic_residuals, start, args=(m, p, mf, k, d, j),
                 full_output=True, xtol=1e-13)
    x, info, ier, msg = sol
    return np.asarray(x), info, ier, msg


def _solve_relatedness(f, m, p, k, d, j, r_init):
    sol = fsolve(_relatedness_residuals, r_init, args=(f, m, p, k, d, j),
                 full_output=True, xtol=1e-13)
    x, info, ier, msg = sol
    return np.asarray(x), info, ier, msg


def _solve_rv(f, m, p, mf, k, d, j):
    sol = fsolve(_rv_residuals, _V_INIT, args=(f, m, p, mf, k, d, j),
                 full_output=True, xtol=1e-13)
    x, info, ier, msg = sol
    return np.asarray(x), info, ier, msg


def sol(m, p, mf, k, d, j):
    """Demographic equilibrium + relatedness (Wolfram `sol`; req init = 0)."""
    f, *_ = _solve_demography(m, p, mf, k, d, j)
    r, *_ = _solve_relatedness(f, m, p, k, d, j, np.zeros(8))
    return f, r


def fullsol(m, p, mf, k, d, j):
    """Demographic equilibrium + relatedness + reproductive values
    (Wolfram `fullsol`; req init = 0.5)."""
    f, *_ = _solve_demography(m, p, mf, k, d, j)
    r, *_ = _solve_relatedness(f, m, p, k, d, j, np.full(8, 0.5))
    v, *_ = _solve_rv(f, m, p, mf, k, d, j)
    return f, r, v


def upd(state, f_init=None, return_f=False):
    """One best-response iteration (Wolfram `upd`).
    state = (m, p, mf, k, err, a, d, j).  req init = 0 (as in the .wl `upd`).

    Optional, non-breaking extensions for branch tracking:
      f_init   : warm-start guess for the demographic solve (default None ->
                 the exact Wolfram starting point _F_INIT).
      return_f : if True, also return the solved demographic vector f, so a
                 caller can thread it as the next iteration's f_init.
    With the defaults (f_init=None, return_f=False) the behaviour is IDENTICAL
    to the verified Wolfram port."""
    m, p, mf, k, err, a, d, j = state
    f, *_ = _solve_demography(m, p, mf, k, d, j, f_init=f_init)
    r, *_ = _solve_relatedness(f, m, p, k, d, j, np.zeros(8))
    v, *_ = _solve_rv(f, m, p, mf, k, d, j)
    d_br, j_br = _best_response(v, r, a, err)
    # damping eta = 0.1: new = 0.9 old + 0.1 best-response; boundaries d9=1, j9=0
    d_new = list(0.9 * np.asarray(d[:8]) + 0.1 * np.asarray(d_br)) + [1.0]
    j_new = list(0.9 * np.asarray(j[:9]) + 0.1 * np.asarray(j_br)) + [0.0]
    new_state = (m, p, mf, k, err, a, np.asarray(d_new), np.asarray(j_new))
    if return_f:
        return new_state, f
    return new_state


def run(m, p, mf, k, err, a, d, j, nit):
    """Iterate `upd` nit times then report final strategies + equilibria
    (Wolfram `run`).  Returns dict with d, j, f, r, v."""
    state = (np.asarray(m, float), np.asarray(p, float), float(mf), float(k),
             float(err), float(a), np.asarray(d, float), np.asarray(j, float))
    for _ in range(nit):
        state = upd(state)
    m, p, mf, k, err, a, d, j = state
    f, r = sol(m, p, mf, k, d, j)
    _, _, v = fullsol(m, p, mf, k, d, j)
    return dict(d=d, j=j, f=f, r=r, v=v)


# ============================================================
# DEFAULT (BASELINE) PARAMETERS  (Wolfram mtest/ptest/dtest/jtest)
# ============================================================

def defaults():
    mtest = np.array([1.0] * 9)
    ptest = np.array([0.2 * (n ** 1.5) * ((9 - n) ** 0.75) for n in range(1, 10)])
    dtest = np.array([0.5 if n < 9 else 1.0 for n in range(1, 10)])
    jtest = np.array([1.0 if n < 9 else 0.0 for n in range(0, 10)])
    return mtest, ptest, dtest, jtest


# ============================================================
# DIAGNOSTICS  (independent of the solver: used for verification)
# ============================================================

def transition_rates(f, m, p, k, d, j):
    """Per-territory birth-death transition rates for group size.
    up[n]   = rate  n -> n+1  (n = 0..8)
    down[n] = rate  n -> n-1  (n = 1..9)
    The chain is a birth-death chain (steps of +/-1 only), so at the
    stationary distribution detailed balance must hold:
        f[n] * up[n] == f[n+1] * down[n+1].
    """
    fd = f[10]
    up = np.zeros(9)    # up[n] for n=0..8
    down = np.zeros(10) # down[n] for n=1..9 (index 0 unused)
    # n = 0: empty territory, growth only via floater immigration
    up[0] = fd * j[0] * k
    for n in range(1, 9):       # n = 1..8
        up[n] = (1 - d[n - 1]) * p[n - 1] + fd * j[n] * k
    for n in range(1, 10):      # n = 1..9
        down[n] = n * m[n - 1]
    return up, down


def net_growth(f, m, p, k, d, j):
    """g[n] = up[n] - down[n] for n = 0..8 (drift of a size-n territory)."""
    up, down = transition_rates(f, m, p, k, d, j)
    g = np.array([up[n] - down[n + 1] for n in range(9)])
    return g
