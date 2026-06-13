# Dual benefits theory — model code

Code accompanying:

> Shen, S.-F. & Rubenstein, D. R. *The ecology and evolution of cooperative
> breeding: linking dual benefits theory and inclusive fitness theory.*
> (*Science Advances*, submitted).

This repository reproduces the model-based figures in the paper. All quantitative
results come from a single demographically explicit, inclusive-fitness model of
group formation (the formal model of Shen, Liu & Rubenstein, *The evolution of
group size and kin structure in complex societies*, cited as ref. 1 in the
Supplementary Materials). The figure scripts call that one engine; nothing is
re-derived independently.

## What each file does

| File | Role | Produces |
|------|------|----------|
| `iomodel.py` | **Core engine.** Faithful Python port of the Wolfram Mathematica model `Appendix_S1_GroupSizeModel.wl`. Solves the demographic (`f`), relatedness (`r`) and reproductive-value (`v`) equilibria and the best-response dispersal/joining strategies. Every expression is transcribed verbatim from the Wolfram source. | (library, imported by the others) |
| `rdca.py` | **RD–CA reparameterization layer.** Builds the vital rates `p(n)`, `m(n)` from mechanistic resource-defense (RD) and collective-action (CA) primitives and exposes branch-tracking and bistability probes. Imports `iomodel`. | (library) |
| `verify_core.py` | **Correctness harness.** Independently checks equilibrium residuals (machine precision), detailed balance, feasibility, the eco-evolutionary fixed point, the qualitative manuscript predictions, and that the closed-form best response equals the verbatim port. | console report |
| `make_box1_fig.py` | **Box 1 figure.** Per-capita performance `w(n)` and the insider acceptance condition under RD vs CA ecology (insider-controlled membership). Reproduces the model means n ≈ 2.1 (RD) and n ≈ 8.2 (CA). | `box1_fig.pdf` / `.svg` / `.png` |
| `make_figS1_box3.py` | **Box 3 figure.** Allee-type benefit vs crowding cost, emergent vs assumed relatedness (insider vs outsider control), and the net-growth drift with small-stable / unstable / large-stable equilibria and the bimodal group-size distribution. | `figS1_box3.pdf` / `.svg` / `.png` |

## Requirements

Python 3.9+ and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

(`numpy`, `scipy`, `matplotlib`.)

## How to run

From inside this directory:

```bash
python3 verify_core.py            # confirm the engine reproduces all checks
python3 make_box1_fig.py          # Box 1 figure
python3 make_figS1_box3.py        # Box 3 figure
```

Each figure script writes its outputs to the current directory. The Box figures
emit editable vector files (text is preserved, `fonttype 42`), suitable for final
adjustment in Illustrator.

## Notes

- `make_box1_fig.py` and `make_figS1_box3.py` require `iomodel.py` (and, for the
  Box 3 figure, `rdca.py`) in the same directory.
- Reproductive skew is not an explicit state variable in the model; within-group
  fecundity is equal. Skew predictions in the paper are inferred from the derived
  kin structure, as described in the Supplementary Materials.

## License

Released under the MIT License (see `LICENSE`).
