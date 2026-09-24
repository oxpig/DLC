# Dropout LayerNorm Correction (DLC)

Standalone reproduction of the dropout→LayerNorm expectation-gap correction
across ten protein structure models: ABodyBuilder3, FlashABB, Ibex,
NbForge, ESMFold, OpenFold, Genie(1/2/3), and QuickBind, plus two negative controls,
ABodyBuilder2 and NanoBodyBuilder2, which were finetuned without dropout.

## Background

Inverted dropout is per-channel unbiased at eval time (`E[dropout(x)] = x`),
but a LayerNorm immediately downstream normalizes using *live* statistics of
its input, which are shared across channels. Because dropout's zeroing
changes those live per-position statistics, `E[LayerNorm(dropout(x))] !=
LayerNorm(x)` even though the dropout itself is individually unbiased. All
models here have at least one dropout→LayerNorm pair in their structure
module. `correction/theory_c.py` implements the
closed-form first-order correction and a small hook-based installer that
applies it at eval time without retraining.

```
c(mu, sigma, q) = sqrt(q / (1 + p*(mu/sigma)^2))          # undamped
damped_c(rho2, q) = sqrt(q / (q + p*rho2))                 # residual-bypass sites
corrected = c * layernorm_output + (1 - c) * layernorm.bias
```

where `q` is the dropout keep probability (`1 - drop_rate`), and `mu`/`sigma`
are the pre-dropout activation's live per-position channel mean/std.

## Repo layout

```
correction/theory_c.py   the correction formula + hook installer, model-agnostic
data/metrics.py          Kabsch alignment, whole-sequence and CDR-H3 RMSD
data/features.py         relative-position pair features shared by ABB3/Ibex/NbForge
data/prepare_splits.py   builds per-structure feature files from a local SAbDab2 copy
data/*_pdb_codes.txt     the exact PDB(+chain) codes used, published for reproducibility
models/*.py               one file per model: load_model() + correction_sites()
scripts/common.py        shared per-model evaluation logic used by the table scripts
scripts/run_*_table.py   reproduces the paper's headline/nanobody RMSD tables
requirements/*.txt        one file per model (their dependencies conflict - use separate envs)
```

## Setup

1. Pick a model, create its environment: `pip install -r requirements/<model>.txt`
   (see requirements/nbforge.txt and requirements/genie3.txt for the two
   models not on PyPI - clone and `pip install -e` instead).
2. Download that model's public checkpoint (see the reference URL in the
   corresponding `models/<model>.py` docstring).
3. Prepare structures: download SAbDab2's public ML-data release
   (https://sabdab2.opig.stats.ox.ac.uk - `ab_split.csv`, `ab_split_sd.csv`,
   and the `.cif` files they reference) into a local directory, then run
   ```
   pip install -r requirements/data.txt
   python data/prepare_splits.py --sabdab2-dir <path> --out-dir data/structures
   ```
4. Run the table script for that model, e.g.:
   ```
   python scripts/run_nanobody_table.py --model abb3 --checkpoint <path/to/checkpoint.ckpt>
   ```

Ibex additionally needs precomputed per-structure ESM-C embeddings (its
protein-language-model embedding and `openfold` don't coexist in one
environment - compute embeddings separately with the ESM-C SDK, one .pt
tensor per structure). OpenFold additionally needs a real per-sequence MSA
per structure (e.g. fetched from the public ColabFold MMseqs2 API) - it
folds close to randomly on a single-sequence "MSA", unlike the
antibody-specific models. See `scripts/run_nanobody_table.py`'s docstring
for the exact flags.
