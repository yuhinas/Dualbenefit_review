# Dual benefits theory: model code

Code accompanying:

> Shen, S.-F., Liu, M. & Rubenstein, D. R. *The ecology and evolution of cooperative
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
| `rdca.py` | **RD–CA reparameterization layer.** Builds the vital rates `p(n)`, `m(n)` from mechanistic resource-defense (RD) and collective-action (CA) primitives, exposes branch-tracking (`run_branch`, `run_branch_g`), and maps the breeding system onto the derived kin structure (`kin_structure_indices`). Imports `iomodel`. | (library) |
| `verify_core.py` | **Correctness harness.** Independently checks equilibrium residuals (machine precision), detailed balance, feasibility, the eco-evolutionary fixed point, the qualitative manuscript predictions, and that the closed-form best response equals the verbatim port. | console report |
| `make_box1_fig.py` | **Box 1 figure.** Per-capita performance `w(n)` and the insider acceptance condition under RD vs CA ecology (insider-controlled membership). Uses an illustrative acceptance-geometry parameter set (distinct from the demographic-model endpoints in `rdca.py` / Table S3); it is a schematic of the acceptance condition, not a source of quantitative means. | `box1_fig.pdf` / `.svg` / `.png` |
| `make_figS1_box3.py` | **Box 3 figure.** Three panels, all computed from the demographic model through `rdca.py`: (A) the ecological increment `ψ_eco(n) = w(n+1) − w(n)` at the RD and CA endpoints, which sets the trend in group size; (B) emergent within-group relatedness `r(n)` under insider-leaning vs outsider-leaning (open-admission) control, the latter serving as a dilution null; (C) emergent relatedness across a fecundity gradient under insider control. | `figS1_box3.pdf` / `.svg` / `.png` |

## The RD–CA endpoints

The dual-benefits axis is a single parameter `theta` in `[0, 1]` that interpolates
all vital-rate primitives between two endpoints (`rdca.py`, `RD_PARAMS` / `CA_PARAMS`):

- **RD endpoint** (`theta = 0`): no task synergy (`smax = 0`), so per-capita
  performance declines monotonically with group size; small, kin-structured groups.
- **CA endpoint** (`theta = 1`): strong task synergy that saturates at intermediate
  size plus modest survival augmentation, so per-capita performance rises to an
  interior optimum; larger, more mixed-kin groups. The stationary mode is interior
  (mean group size ≈ 6, with negligible mass at the `n = 9` upper bound), so the
  large-group result is not a truncation artifact of the model's maximum size.

Under insider control, kin-biased admission keeps within-group relatedness elevated
relative to the dilution expectation under random (open) admission; higher fecundity
promotes natal recruitment and tends to sustain higher relatedness in larger groups.

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

Each figure script writes its outputs to the current directory. The figure scripts
emit editable vector files (text is preserved, `fonttype 42`), suitable for final
adjustment in Illustrator.

## Notes

- `make_figS1_box3.py` requires `iomodel.py` and `rdca.py` in the same directory;
  `make_box1_fig.py` requires `iomodel.py`.
- Reproductive skew is not an explicit state variable in the model; within-group
  fecundity is equal. Skew predictions in the paper are inferred from the derived
  kin structure (relatedness and natal-recruitment fraction), as described in the
  Supplementary Materials.
- Mean within-group relatedness is summarised by weighting each group size by its
  stationary frequency, i.e. it is the expected relatedness of a randomly sampled
  group rather than a pair-weighted population average.

## License

Released under the MIT License (see `LICENSE`).
