r"""
make_box1_fig.py
================
Box 1 figure for "The ecology and evolution of cooperative breeding: linking dual
benefits theory and inclusive fitness theory".

Both panels use insider-controlled membership in the formal demographic group-formation
model (iomodel.py, the port of Appendix S1); only the ecology differs, so the contrast is
clean. Two ecologies are each solved to the eco-evolutionary equilibrium at a = 0.1.

The two ecologies here use an ILLUSTRATIVE acceptance-geometry parameter set chosen to
display the per-capita and acceptance curves clearly. These are distinct from the
RD/CA endpoint parameters of the demographically explicit model used for the Box 3 figure
(those are listed in rdca.py and in Table S3 of the Supplementary Materials); the Box 1
figure is a schematic of the acceptance condition, not a source of quantitative means.

  RD-dominated : per-capita fecundity declines with group size (crowding), mortality
                 constant.  ->  per-capita performance w(n) declines.
  CA-dominated : mild fecundity crowding but mortality falls with group size
                 (survival-based group augmentation).  ->  w(n) rises to an interior
                 optimum then falls (positive density dependence).

From each solved model we take the per-capita performance w(n) = (p(n)/n)/m(n), the
emergent within-group relatedness r(n), and the equilibrium mean group size.

Box 1 insider acceptance condition for adding one member at size n
(psi(n) = w(n+1) - w(n); outsider-insider relatedness r_OI; outsider solitary option s):

        psi(n)  +  r_OI ( w(n+1) - s )  >=  0
        \____/     \__________________/
        direct          indirect (kin)

psi(n) also equals the net for a non-kin joiner (r_OI = 0). Under RD psi < 0, so acceptance
needs the kin term and groups stay small and kin-structured. Under CA psi > 0 up to the
per-capita optimum, so adding a member raises every resident's performance and acceptance
is favored directly, even for a non-kin joiner up to the optimum; the kin term extends it
and groups become large.

Requires iomodel.py in the same directory. Outputs editable vector files (text stays
editable): box1_fig.pdf, box1_fig.svg, and a png preview.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import iomodel as M

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

NMAX = 9
n = np.arange(1, NMAX + 1)
mf, k, err = 1.0, 7.0, 0.05
d0, j0 = M.defaults()[2], M.defaults()[3]
A_INSIDER = 0.1                                   # insider-controlled membership (both panels)

def p_RD(nn, b=2.0, alpha=0.35): return nn * (b / (1 + alpha * (nn - 1)))   # crowded fecundity
def m_RD(nn):                    return np.ones_like(nn, float)             # constant mortality
def p_CA(nn, b=2.2, beta=0.18):  return nn * (b / (1 + beta * (nn - 1)))    # mild crowding
def m_CA(nn, m0=1.0, mmin=0.30, Kd=2.0, hd=1.5):                            # survival augmentation
    return mmin + (m0 - mmin) / (1 + ((nn - 1) / Kd) ** hd)                 # hd=1.5 -> w rises from n=1

def solve(p, m):
    R = M.run(m, p, mf, k, err, A_INSIDER, d0, j0, 1000)
    occ = R["f"][1:10] / R["f"][1:10].sum()
    meann = float(sum(i * occ[i - 1] for i in range(1, 10)))
    w = (p / n) / m
    rOI = np.asarray(R["r"], float)               # relatedness at sizes 2..9 -> adding at n=1..8
    return w, rOI, meann

pRD, mRD = p_RD(n), m_RD(n)
pCA, mCA = p_CA(n), m_CA(n)
wRD, rRD, meanRD = solve(pRD, mRD)
wCA, rCA, meanCA = solve(pCA, mCA)
sRD, sCA = 0.85, 2.25                              # outsider solitary option
xg = n[:-1]

def decomp(w, rOI, s):
    psi = np.diff(w)                               # direct term (= non-kin net)
    indir = rOI * (w[1:] - s)                       # indirect kin term
    return psi, indir, psi + indir                  # ..., net acceptance (kin joiner)

psiRD, indRD, netRD = decomp(wRD, rRD, sRD)
psiCA, indCA, netCA = decomp(wCA, rCA, sCA)

def cross_down(x, y):
    for i in range(len(x) - 1):
        if y[i] > 0 >= y[i + 1]:
            return x[i] + y[i] / (y[i] - y[i + 1])
    return x[-1]
def cross_up(x, y):
    for i in range(len(x) - 1):
        if y[i] <= 0 < y[i + 1]:
            return x[i] + (-y[i]) / (y[i + 1] - y[i])
    return None

ukon, kikyo, ai = "#E69B3A", "#5654A2", "#165E83"
nezumi, akane, hanada = "#7B7C7D", "#B7282E", "#2D6D9B"
oitake, shadeRD, shadeCA, sumi = "#6E8B5B", "#F4D9D4", "#D6E6EC", "#2B2B2B"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": sumi, "axes.linewidth": 0.8, "axes.labelcolor": sumi,
    "xtick.color": sumi, "ytick.color": sumi, "text.color": sumi})
fig, ax = plt.subplots(2, 2, figsize=(9.7, 7.1))
(axRDw, axCAw), (axRDg, axCAg) = ax
def style(a):
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    a.set_xlim(0.6, 9.2); a.set_xticks(range(1, 10))

# ===== top row: per-capita performance w(n) =====
def topw(a, w, s, accent, sub, optmark):
    a.axhline(s, color=nezumi, lw=1.0, ls=(0, (5, 3)))
    a.plot(n, w, "-o", color=oitake, lw=2.6, ms=5)
    a.annotate("outside option s", xy=(7.9, s), xytext=(7.9, s + (0.16 if optmark else 0.13)),
               color=nezumi, fontsize=8, ha="center")
    a.text(0.5, 1.01, sub, transform=a.transAxes, ha="center", va="bottom", fontsize=9, color=sumi)
    a.set_ylabel("per-capita performance  w(n) = (p/n)/m"); style(a)
    if optmark:
        no = 1 + int(np.argmax(w))
        a.plot([no], [w[no - 1]], "o", color=accent, ms=9, zorder=5)
        a.annotate("optimum", xy=(no, w[no - 1]), xytext=(no + 1.8, w[no - 1] + 0.01),
                   color=accent, fontsize=8.5, ha="center",
                   arrowprops=dict(arrowstyle="-", color=accent, lw=0.8))
        a.annotate("rises", xy=(2.6, w[1]), color=oitake, fontsize=8.5, ha="center")
        a.annotate("then falls", xy=(8.3, w[7]), color=oitake, fontsize=8.5, ha="center")
topw(axRDw, wRD, sRD, akane, "crowding: per-capita performance declines", False)
axRDw.set_title("RD ecology", color=akane, fontsize=12, pad=14)
topw(axCAw, wCA, sCA, hanada, "positive density dependence: rises then falls", True)
axCAw.set_title("CA ecology", color=hanada, fontsize=12, pad=14)

# ===== bottom row: insider acceptance gain (both insider control) =====
def botg(a, psi, ind, net, s, shade, accent, mean, headline, nonkin_note):
    cdn = cross_down(xg, net)
    a.axvspan(0.6, cdn, color=shade, alpha=0.8, lw=0)
    a.axhline(0, color=sumi, lw=0.8, ls=(0, (4, 3)))
    a.plot(xg, psi, "-o", color=ukon, lw=1.8, ms=4, label=r"direct  $\psi(n)=w_{n+1}-w_n$  (= non-kin net)")
    a.plot(xg, ind, "-o", color=kikyo, lw=1.8, ms=4, label=r"indirect (kin)  $r_{OI}(w_{n+1}-s)$")
    a.plot(xg, net, "-o", color=ai, lw=2.6, ms=5, label="net acceptance (kin joiner)")
    a.plot([cdn], [0], "o", color=accent, ms=8, zorder=5)
    a.annotate("stable size", xy=(cdn, 0), xytext=(cdn - 1.4, 0.18 * np.sign(net[0] + 1e-9) + 0.05),
               color=accent, fontsize=8.5, ha="center",
               arrowprops=dict(arrowstyle="-", color=accent, lw=0.8))
    a.text(0.5, 1.005, headline % mean, transform=a.transAxes, ha="center", va="bottom",
           fontsize=8.6, color=sumi)
    a.set_xlabel("group size  n"); style(a)
    return cdn

axRDg.set_ylabel("insider acceptance gain from one more member")
botg(axRDg, psiRD, indRD, netRD, sRD, shadeRD, akane, meanRD,
     "insiders gate, RD ecology  \u2192  small, kin-structured (model mean n \u2248 %.1f)", None)
axRDg.annotate("non-kin joiner: net = $\\psi$ < 0 (excluded)", xy=(4.0, psiRD[3]),
               xytext=(6.2, -0.33), color=ukon, fontsize=7.6, ha="center",
               arrowprops=dict(arrowstyle="-", color=ukon, lw=0.8))
axRDg.legend(frameon=False, fontsize=7.0, loc="upper right")

cdnCA = botg(axCAg, psiCA, indCA, netCA, sCA, shadeCA, hanada, meanCA,
     "insiders gate, CA ecology  \u2192  large (model mean n \u2248 %.1f)", None)
nopt = 1 + int(np.argmax(wCA))
axCAg.axvline(nopt, color=oitake, lw=0.8, ls=(0, (1, 2)))
axCAg.annotate("$\\psi>0$ up to the optimum:\nnon-kin accepted too", xy=(nopt, 0.0),
               xytext=(6.0, 0.45), color=oitake, fontsize=7.6, ha="center",
               arrowprops=dict(arrowstyle="-", color=oitake, lw=0.8))

for a, lab in [(axRDw, "A"), (axCAw, "B"), (axRDg, "C"), (axCAg, "D")]:
    a.text(-0.13, 1.07, lab, transform=a.transAxes, fontsize=14, fontweight="bold",
           color=sumi, va="top", ha="left")

fig.tight_layout(w_pad=2.0, h_pad=2.2)
fig.savefig("box1_fig.pdf", bbox_inches="tight")
fig.savefig("box1_fig.svg", bbox_inches="tight")
fig.savefig("box1_fig.png", dpi=200, bbox_inches="tight")
print("RD: w peak@n=%d mean=%.2f r=[%.2f..%.2f] stable@%.2f" %
      (1 + int(np.argmax(wRD)), meanRD, rRD.min(), rRD.max(), cross_down(xg, netRD)))
print("CA: w peak@n=%d mean=%.2f r=[%.2f..%.2f] stable@%.2f  psi>0 up to n=%d" %
      (1 + int(np.argmax(wCA)), meanCA, rCA.min(), rCA.max(), cdnCA, np.sum(psiCA > 0)))
print("saved box1_fig.pdf / .svg / .png")
