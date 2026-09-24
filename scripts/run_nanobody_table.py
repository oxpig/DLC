"""Reproduces the paper's nanobody-set result (uncorrected vs corrected
whole-sequence full-backbone RMSD, averaged over data/nanobody_100_pdb_codes.txt)
for one model at a time - each model needs its own environment (see
requirements/), so this is invoked once per model, not as a single run
across all of them.

Structures must already be prepared: run data/prepare_splits.py first.

Usage:
    python scripts/run_nanobody_table.py --model abb3 --checkpoint <path>
    python scripts/run_nanobody_table.py --model ibex --checkpoint <path> --plm-embeddings-dir <dir>
    python scripts/run_nanobody_table.py --model nbforge --checkpoint <path>
    python scripts/run_nanobody_table.py --model esmfold
    python scripts/run_nanobody_table.py --model openfold --checkpoint <path> --msa-dir <dir>
    python scripts/run_nanobody_table.py --model nanobodybuilder2 --checkpoint <weights cache dir>

Ibex needs precomputed per-structure ESM-C embeddings (one .pt CA-ordered
tensor per structure, named <structure>.pt) since ESM-C and openfold don't
coexist in one environment - compute them separately with the ESM-C SDK.
OpenFold needs precomputed real MSAs (one .a3m per structure, named
<structure>.a3m, e.g. fetched from the public ColabFold MMseqs2 API) since
it folds close to randomly on a single-sequence "MSA".

nanobodybuilder2 is architecturally dropout-free (see
models/nanobodybuilder2.py) - there is no correction to apply, so its row
is plain accuracy only (uncorrected == corrected by construction).
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from scripts.common import load_structures, run_model, print_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["abb3", "ibex", "nbforge", "esmfold", "openfold", "nanobodybuilder2"])
    parser.add_argument("--checkpoint", help="Required for abb3/ibex/nbforge/openfold; for nanobodybuilder2, "
                                              "a local directory to cache weights auto-downloaded from Zenodo")
    parser.add_argument("--structures-dir", default=HERE.parent / "data" / "structures", type=Path)
    parser.add_argument("--plm-embeddings-dir", help="Required for ibex")
    parser.add_argument("--msa-dir", help="Required for openfold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = load_structures(HERE.parent / "data" / "nanobody_100_pdb_codes.txt", args.structures_dir)
    print(f"n structures: {len(items)}\n")

    results = run_model(
        args.model, args.checkpoint, items, device,
        plm_embeddings_dir=args.plm_embeddings_dir, msa_dir=args.msa_dir,
    )
    print_summary(args.model, "nanobody set", results)


if __name__ == "__main__":
    main()
