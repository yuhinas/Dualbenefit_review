"""
verify_core.py
==============
Re-runnable correctness harness for the faithful port (iomodel.py).

Checks, with independent cross-validation wherever possible:
  [1] residuals of all three equilibrium systems are at machine precision
  [2] detailed balance  f[n]*up[n] == f[n+1]*down[n+1]  (INDEPENDENT encoding
      of the birth-death chain, not copied from the residual equations)
  [3] basic feasibility: sum f == 1, f >= 0, fd >= 0, r in [0,1], v > 0,
      reproductive-value normalization == 1
  [4] eco-evolutionary fixed point under outsider control (a=0.9) converges
  [5] qualitative manuscript predictions reproduced
  [6] warm-start vs cold-start give the SAME root in the unique-equilibrium
      regime (confirms the optional f_init change is non-breaking)

Run:  python3 verify_core.py
"""
import numpy as np
import iomodel as M

TOL = 1e-9
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


m, p, d, j = M.defaults()
mf, k, err = 1.0, 7.0, 0.05

print("=" * 64)
print("[1-3] single solve at initial strategies (d=0.5, j=1)")
f, info, ier, msg = M._solve_demography(m, p, mf, k, d, j)
res_f = np.max(np.abs(M._demographic_residuals(f, m, p, mf, k, d, j)))
check("demographic residual < tol", res_f < TOL, f"max|res|={res_f:.1e}")
check("sum f == 1", abs(f[:10].sum() - 1) < TOL, f"sum={f[:10].sum():.12f}")
check("f >= 0 and fd >= 0", (f >= -TOL).all(), f"min={f.min():.2e}")

up, down = M.transition_rates(f, m, p, k, d, j)
db = np.max(np.abs([f[n] * up[n] - f[n + 1] * down[n + 1] for n in range(9)]))
check("detailed balance (independent check) < tol", db < TOL, f"max|db|={db:.1e}")

r, *_ = M._solve_relatedness(f, m, p, k, d, j, np.zeros(8))
res_r = np.max(np.abs(M._relatedness_residuals(r, f, m, p, k, d, j)))
check("relatedness residual < tol", res_r < TOL, f"max|res|={res_r:.1e}")
check("relatedness in [0,1]", ((r >= -TOL) & (r <= 1 + TOL)).all(),
      f"min={r.min():.3f} max={r.max():.3f}")

v, *_ = M._solve_rv(f, m, p, mf, k, d, j)
res_v = np.max(np.abs(M._rv_residuals(v, f, m, p, mf, k, d, j)))
check("reproductive-value residual < tol", res_v < TOL, f"max|res|={res_v:.1e}")
check("reproductive values > 0", (v > 0).all(), f"min={v.min():.3f}")
num = sum(n * f[n] * v[n - 1] for n in range(1, 10)) + f[10] * v[9]
den = sum(n * f[n] for n in range(1, 10)) + f[10]
check("RV-weighted mean == 1", abs(num / den - 1) < TOL, f"mean={num/den:.12f}")

print("\n[4] eco-evolutionary fixed point, outsider control a=0.9, nit=1000")
R9 = M.run(m, p, mf, k, err, 0.9, d, j, 1000)
st = (m, p, mf, k, err, 0.9, R9['d'], R9['j'])
st2 = M.upd(st)
gap = max(np.max(np.abs(st2[6] - R9['d'])), np.max(np.abs(st2[7] - R9['j'])))
check("strategies are a fixed point (one more upd ~ no change)", gap < 1e-6,
      f"max change={gap:.1e}")
res_f9 = np.max(np.abs(M._demographic_residuals(
    np.append(R9['f'][:10], R9['f'][10]), m, p, mf, k, R9['d'], R9['j'])))
check("equilibrium residual at final strategies < tol", res_f9 < TOL, f"max|res|={res_f9:.1e}")

print("\n[5] qualitative predictions (manuscript)")
mostprod = 1 + int(np.argmax([p[n - 1] / n for n in range(1, 10)]))
check("most productive size == 4", mostprod == 4, f"={mostprod}")

R1 = M.run(m, p, mf, k, err, 0.1, d, j, 1000)   # insider control
occ1 = R1['f'][1:10] / R1['f'][1:10].sum()
occ9 = R9['f'][1:10] / R9['f'][1:10].sum()
mean1 = sum(n * occ1[n - 1] for n in range(1, 10))
mean9 = sum(n * occ9[n - 1] for n in range(1, 10))
check("outsider control -> larger groups than insider control", mean9 > mean1,
      f"mean(a=.9)={mean9:.2f} > mean(a=.1)={mean1:.2f}")
rbar1 = np.average(R1['r'], weights=R1['f'][2:10])
rbar9 = np.average(R9['r'], weights=R9['f'][2:10])
check("insider control -> higher relatedness than outsider control", rbar1 > rbar9,
      f"rbar(a=.1)={rbar1:.2f} > rbar(a=.9)={rbar9:.2f}")
check("insider control: relatedness rises with size (r9 > r2)",
      R1['r'][-1] > R1['r'][0], f"r2={R1['r'][0]:.2f} -> r9={R1['r'][-1]:.2f}")
# net-growth sign structure
check("insider control: net-growth drift crosses zero (a stable interior size exists)",
      np.any(np.diff(np.sign(M.net_growth(R1['f'], m, p, k, R1['d'], R1['j']))) != 0),
      "stationary group-size distribution has an interior mode")

print("\n[6] warm-start vs cold-start, unique-equilibrium (baseline) regime")
f_cold, *_ = M._solve_demography(m, p, mf, k, d, j)               # default Wolfram init
perturbed = f_cold * np.linspace(0.5, 1.5, 11)                    # very different seed
f_warm, *_ = M._solve_demography(m, p, mf, k, d, j, f_init=perturbed)
check("warm and cold start reach the same root", np.max(np.abs(f_warm - f_cold)) < 1e-8,
      f"max diff={np.max(np.abs(f_warm-f_cold)):.1e}")

print("\n[7] closed-form best response == verbatim Wolfram port")
_rng = np.random.default_rng(12345)
_maxd = _maxj = 0.0
for _ in range(3000):
    _v = _rng.uniform(0.2, 2.0, 10)
    _r = _rng.uniform(0.0, 1.0, 8)
    _a = _rng.uniform(0.0, 1.0)
    _err = _rng.uniform(0.02, 0.5)
    _d0, _j0 = M._best_response(_v, _r, _a, _err)
    _d1, _j1 = M._best_response_closed(_v, _r, _a, _err)
    _maxd = max(_maxd, np.max(np.abs(np.array(_d1) - np.array(_d0))))
    _maxj = max(_maxj, np.max(np.abs(np.array(_j1) - np.array(_j0))))
check("dispersal d1..d8: closed form matches verbatim port", _maxd < 1e-12,
      f"max|d_closed - d_code|={_maxd:.1e} over 3000 random (v,r,a,err)")
check("joining  j0..j8: closed form matches verbatim port", _maxj < 1e-12,
      f"max|j_closed - j_code|={_maxj:.1e} over 3000 random (v,r,a,err)")

print("=" * 64)
print("OVERALL:", "ALL CHECKS PASSED" if ok_all else "*** SOME CHECKS FAILED ***")
