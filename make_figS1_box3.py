"""
make_figS1_box3.py
==================
Box 3 figure (formerly SI Figure S1), rebuilt to reflect the demography-driven
kin-structure story. Every demographic quantity is computed by the verified
iomodel engine through the rdca reparameterization layer; nothing is reimplemented.

  (A) The ecological increment psi_eco(n) = w(n+1) - w(n), evaluated at fixed relatedness
      and environment, at the RD endpoint (negative:
      crowding on a finite resource) and the CA endpoint (positive over a range:
      positive density dependence). This sets the trend in group SIZE, not kin structure.
  (B) Emergent within-group relatedness r(n) under insider vs outsider control at a
      fixed intermediate ecology. Outsider control (open admission) is the dilution
      null; insider control (kin-biased admission) lies ABOVE it at every size. The
      robust signature is this elevation, not the absolute sign of the slope.
  (C) Emergent r(n) under insider control across a fecundity gradient g: low fecundity
      keeps small groups dependent on unrelated immigrants (lower r); high fecundity
      lets groups grow through natal recruitment (higher r relative to the null).

Requires iomodel.py and rdca.py in the same directory.
Outputs editable vector files: figS1_box3.pdf, figS1_box3.svg, and a png preview.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rdca as RC
import warnings
# Suppress only benign numerical RuntimeWarnings (e.g. transient overflow/invalid
# values inside the iterative solve); all results are independently checked to
# machine precision in verify_core.py. Other warnings are left visible.
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

N = np.arange(1, 10)
NN = np.arange(2, 10)                 # sizes where within-group relatedness is defined
A_INSIDER, A_OUTSIDER = 0.3, 0.9      # insider-leaning vs outsider control
THETA_MID = 0.3                        # intermediate ecology for panels B, C
G_MID = 1.3                            # fecundity for panel B
G_LIST = [1.0, 1.3, 1.6, 2.0]          # fecundity gradient for panel C

# ---- compute everything through the verified engine ----
# Panel A: per-capita increment at the two endpoints
def psi_endpoint(theta):
    m, p = RC.vital_rates(theta)
    w = (p / N) / m
    return np.diff(w)                  # psi(n) = w(n+1) - w(n), length 8 (n=1..8)
psiRD = psi_endpoint(0.0)
psiCA = psi_endpoint(1.0)

# Panel B: insider vs outsider emergent relatedness at fixed ecology
Rin, _ = RC.run_branch_g(THETA_MID, A_INSIDER, g=G_MID, seed="large", nit=1200)
Rout, _ = RC.run_branch_g(THETA_MID, A_OUTSIDER, g=G_MID, seed="large", nit=1200)
r_in, r_out = np.asarray(Rin["r"]), np.asarray(Rout["r"])     # n=2..9

# Panel C: emergent relatedness across fecundity (insider control)
rC = []
for g in G_LIST:
    Rg, _ = RC.run_branch_g(THETA_MID, A_INSIDER, g=g, seed="large", nit=1200)
    rC.append(np.asarray(Rg["r"]))

# ---- Japanese palette (same as Box 1 figure) ----
ukon, kikyo, ai = "#E69B3A", "#5654A2", "#165E83"
nezumi, akane, hanada = "#7B7C7D", "#B7282E", "#2D6D9B"
oitake, shadeRD, shadeCA, sumi = "#6E8B5B", "#F4D9D4", "#D6E6EC", "#2B2B2B"
# fecundity gradient colors (light -> dark teal/indigo)
g_colors = ["#BFD3C1", "#7FA9A0", "#3D7E91", "#165E83"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": sumi, "axes.linewidth": 0.8, "axes.labelcolor": sumi,
    "xtick.color": sumi, "ytick.color": sumi, "text.color": sumi})
fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.5))
aA, aB, aC = ax
def style(a):
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    a.set_xlim(0.6, 9.3); a.set_xticks(range(1, 10))

# ===== (A) per-capita increment psi(n) at endpoints =====
xg = N[:-1]                            # n = 1..8
aA.axhline(0, color=nezumi, lw=1.0, ls=(0, (5, 3)))
aA.plot(xg, psiRD, "-o", color=akane, lw=2.4, ms=4, label="RD endpoint (crowding)")
aA.plot(xg, psiCA, "-o", color=hanada, lw=2.4, ms=4, label="CA endpoint (synergy)")
# shade where CA psi > 0
pos = psiCA > 0
if pos.any():
    lo = xg[pos][0] - 0.4
    hi = xg[pos][-1] + 0.4
    aA.axvspan(lo, hi, color=shadeCA, alpha=0.6, lw=0)
aA.set_xlabel("group size  n"); aA.set_ylabel(r"ecological increment  $\psi_{\mathrm{eco}}(n)=w_{n+1}-w_n$")
aA.legend(frameon=False, fontsize=8, loc="upper right")
aA.text(0.5, 1.02, r"RD: $\psi_{\mathrm{eco}}<0$ (smaller groups);  CA: $\psi_{\mathrm{eco}}>0$ (larger groups)",
        transform=aA.transAxes, ha="center", va="bottom", fontsize=8.2, color=sumi)
style(aA)

# ===== (B) insider vs outsider relatedness; outsider = dilution null =====
aB.plot(NN, r_out, "--s", color=nezumi, lw=2.0, ms=4,
        label="outsider-leaning (open-admission null)")
aB.plot(NN, r_in, "-o", color=ai, lw=2.6, ms=5,
        label="insider-leaning (kin-biased)")
aB.fill_between(NN, r_out, r_in, where=(r_in >= r_out), color=shadeCA, alpha=0.5, lw=0)
aB.set_ylim(0, 1.0)
aB.set_xlabel("group size  n"); aB.set_ylabel("within-group relatedness  r(n)")
aB.legend(frameon=False, fontsize=8, loc="center right")
aB.text(0.5, 1.02, "insider control sits above the dilution null",
        transform=aB.transAxes, ha="center", va="bottom", fontsize=8.4, color=sumi)
style(aB)

# ===== (C) emergent relatedness across fecundity (insider) =====
for g, r, c in zip(G_LIST, rC, g_colors):
    aC.plot(NN, r, "-o", color=c, lw=2.2, ms=4, label=f"g = {g:.1f}")
aC.set_ylim(0, 1.0)
aC.set_xlabel("group size  n"); aC.set_ylabel("within-group relatedness  r(n)")
aC.legend(frameon=False, fontsize=8, loc="lower right", title="fecundity", title_fontsize=8)
aC.text(0.5, 1.02, "higher fecundity sustains relatedness in larger groups",
        transform=aC.transAxes, ha="center", va="bottom", fontsize=8.4, color=sumi)
style(aC)

for a, lab in [(aA, "A"), (aB, "B"), (aC, "C")]:
    a.text(-0.16, 1.10, lab, transform=a.transAxes, fontsize=14, fontweight="bold",
           color=sumi, va="top", ha="left")

fig.tight_layout(w_pad=2.2)
fig.savefig("figS1_box3.pdf", bbox_inches="tight")
fig.savefig("figS1_box3.svg", bbox_inches="tight")
fig.savefig("figS1_box3.png", dpi=200, bbox_inches="tight")

# ---- report numbers for the SI / caption ----
mRD = RC.run_branch_g(0.0, A_INSIDER, g=1.0, seed="large", nit=1200)[1]["mean"]
mCA = RC.run_branch_g(1.0, A_INSIDER, g=1.0, seed="large", nit=1200)[1]["mean"]
print("endpoint means (insider a=%.1f, g=1.0): RD n=%.2f  CA n=%.2f" % (A_INSIDER, mRD, mCA))
print("panel B insider r(n):", " ".join("%.2f" % x for x in r_in))
print("panel B outsider r(n):", " ".join("%.2f" % x for x in r_out))
print("panel B insider>outsider at all n:", bool(np.all(r_in > r_out)))
print("panel C r(9) by g:", " ".join("g%.1f=%.2f" % (g, r[-1]) for g, r in zip(G_LIST, rC)))
print("saved figS1_box3.{pdf,svg,png}")
