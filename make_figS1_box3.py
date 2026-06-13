"""
make_figS1_box3.py
==================
Box 3 figure (formerly SI Figure S1), rebuilt as ONE combined image in the same
Japanese-colour style as the Box 1 figure. Every demographic quantity is from the
verified iomodel engine via the rdca reparameterization layer; nothing is reimplemented.

  (A) per-capita increment components: Allee-type CA benefit B_A(n) and a general
      cost C(n) (additive intuition, Box 3 canonical parameters)
  (B) emergent within-group relatedness r(n) under insider vs outsider control,
      against the assumed monotonic dilution (assumed -> emergent)
  (C) net-growth drift g(n) with the small-stable / unstable / large-stable
      equilibria and the resulting bimodal group-size distribution

Requires iomodel.py and rdca.py in the same directory.
Outputs editable vector files: figS1_box3.pdf, figS1_box3.svg, and a png preview.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rdca as RC
import iomodel as M

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

K, TH = 7.0, 0.25
N = np.arange(1, 10)

# ---- compute (all via iomodel through rdca) ----
m, p = RC.vital_rates(TH)                                  # multiplicative model (panels B,C)
Ri, _ = RC.run_branch(TH, 0.1, seed="large", nit=1200)     # insider control
Ro, _ = RC.run_branch(TH, 0.9, seed="large", nit=1200)     # outsider control
r_assumed = RC.box3_assumed_relatedness()                  # assumed dilution, n=2..9
occ = Ri["f"][1:10] / Ri["f"][1:10].sum()
g = np.asarray(M.net_growth(Ri["f"], m, p, K, Ri["d"], Ri["j"]))   # drift, sizes 0..8
_, _, comp = RC.vital_rates_box3()                         # additive B_A/C (panel A)
BA, C = comp["BA"], comp["C"]

def zero_crossings(g):
    out = []
    for i in range(1, len(g) - 1):
        if g[i] * g[i + 1] < 0:
            root = i + g[i] / (g[i] - g[i + 1])
            out.append((root, "stable" if (g[i] > 0 and g[i + 1] < 0) else "unstable"))
    return out
roots = zero_crossings(g)

# ---- Japanese palette (same as Box 1 figure) ----
ukon, kikyo, ai = "#E69B3A", "#5654A2", "#165E83"
nezumi, akane, hanada = "#7B7C7D", "#B7282E", "#2D6D9B"
oitake, shadeRD, shadeCA, sumi = "#6E8B5B", "#F4D9D4", "#D6E6EC", "#2B2B2B"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": sumi, "axes.linewidth": 0.8, "axes.labelcolor": sumi,
    "xtick.color": sumi, "ytick.color": sumi, "text.color": sumi})
fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.5))
aA, aB, aC = ax
def style(a):
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    a.set_xlim(0.6, 9.3); a.set_xticks(range(1, 10))

# ===== (A) B_A(n) and C(n) components =====
psi = BA - C
xc = [N[i] + (psi[i]) / (psi[i] - psi[i+1]) for i in range(len(psi)-1) if psi[i]*psi[i+1] < 0]
if len(xc) >= 2:
    aA.axvspan(xc[0], xc[1], color=shadeCA, alpha=0.7, lw=0)
aA.plot(N, BA, "-o", color=hanada, lw=2.4, ms=4, label=r"collective-action benefit  $B_A(n)$")
aA.plot(N, C, "-o", color=akane, lw=2.4, ms=4, label=r"crowding/conflict cost  $C(n)$")
aA.set_xlabel("group size  n"); aA.set_ylabel("per-capita increment component")
aA.legend(frameon=False, fontsize=8, loc="upper left")
aA.text(0.5, 1.02, r"$\psi(n)=B_A(n)-C(n)>0$ over the CA window", transform=aA.transAxes,
        ha="center", va="bottom", fontsize=8.4, color=sumi)
style(aA)

# ===== (B) emergent vs assumed relatedness =====
nn = np.arange(2, 10)
aB.plot(nn, Ri["r"], "-o", color=ai, lw=2.4, ms=4, label="insider control (derived)")
aB.plot(nn, Ro["r"], "-s", color=akane, lw=2.4, ms=4, label="outsider control (derived)")
aB.plot(nn, r_assumed, "--", color=nezumi, lw=1.8, label="assumed dilution")
aB.set_xlabel("group size  n"); aB.set_ylabel("within-group relatedness  r(n)")
aB.set_ylim(0, 1.0); aB.legend(frameon=False, fontsize=8, loc="center right")
aB.text(0.5, 1.02, "kin retained vs diluted (emergent)", transform=aB.transAxes,
        ha="center", va="bottom", fontsize=8.4, color=sumi)
style(aB)

# ===== (C) net-growth drift + bimodal distribution =====
aCb = aC.twinx()
aCb.bar(N, occ, color=nezumi, alpha=0.22, width=0.7)
aCb.set_ylim(0, occ.max() * 2.4); aCb.set_yticks([])
aCb.spines["top"].set_visible(False)
sizes = np.arange(1, 9)
aC.axhline(0, color=nezumi, lw=1.0, ls=(0, (5, 3)))
aC.plot(sizes, g[1:9], "-o", color=ai, lw=2.4, ms=4, zorder=5)
for root, kind in roots:
    if kind == "stable":
        aC.plot(root, 0, "o", ms=9, color=sumi, zorder=6)
        aC.annotate(r"$n^{\ast}$", (root, 0), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=10, color=sumi)
    else:
        aC.plot(root, 0, "o", ms=9, mfc="white", mec=sumi, mew=1.6, zorder=6)
        aC.annotate(r"$n^{\dagger}$", (root, 0), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=10, color=sumi)
aC.set_xlabel("group size  n"); aC.set_ylabel("consensus drift  g(n)")
aC.text(0.5, 1.02, "small-stable / unstable / large-stable", transform=aC.transAxes,
        ha="center", va="bottom", fontsize=8.4, color=sumi)
style(aC)

for a, lab in [(aA, "A"), (aB, "B"), (aC, "C")]:
    a.text(-0.16, 1.10, lab, transform=a.transAxes, fontsize=14, fontweight="bold",
           color=sumi, va="top", ha="left")

fig.tight_layout(w_pad=2.2)
fig.savefig("figS1_box3.pdf", bbox_inches="tight")
fig.savefig("figS1_box3.svg", bbox_inches="tight")
fig.savefig("figS1_box3.png", dpi=200, bbox_inches="tight")
print("roots:", [(round(r,2),k) for r,k in roots])
print("insider r:", " ".join(f"{x:.2f}" for x in Ri["r"]))
print("outsider r:", " ".join(f"{x:.2f}" for x in Ro["r"]))
print("saved figS1_box3.{pdf,svg,png}")
