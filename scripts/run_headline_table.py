"""Reproduces the paper's headline result (uncorrected vs corrected
whole-sequence full-backbone RMSD, averaged over data/headline_34_pdb_codes.txt
- paired heavy+light Fv structures) for one model at a time. Same usage
pattern as run_nanobody_table.py - see that script's docstring for
per-model setup requirements.

OpenFold isn't supported here yet: a paired headline-table row needs the
AlphaFold-Gap trick (concatenated sequence, residue-index offset at the
chain break, block-diagonal MSA), which isn't implemented in this repo -
see scripts/common.py's eval_openfold docstring.

abb2 (ABodyBuilder2) is architecturally dropout-free (see models/abb2.py) -
there is no correction to apply, so its row is plain accuracy only
(uncorrected == corrected by construction).
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
    parser.add_argument("--model", required=True, choices=["abb3", "ibex", "nbforge", "esmfold", "flashabb", "abb2"])
    parser.add_argument("--checkpoint", help="Required for abb3/ibex/nbforge (flashabb's checkpoint is bundled in the package); "
                                              "for abb2, a local directory to cache weights auto-downloaded from Zenodo")
    parser.add_argument("--structures-dir", default=HERE.parent / "data" / "structures", type=Path)
    parser.add_argument("--plm-embeddings-dir", help="Required for ibex")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = load_structures(HERE.parent / "data" / "headline_34_pdb_codes.txt", args.structures_dir)
    print(f"n structures: {len(items)}\n")

    results = run_model(args.model, args.checkpoint, items, device, plm_embeddings_dir=args.plm_embeddings_dir)
    print_summary(args.model, "headline set", results)


if __name__ == "__main__":
    main()
