"""Reproduces the paper's QuickBind result (Table: ligand-atom RMSD,
uncorrected vs. corrected, on the PoseBusters Benchmark set - 427
structures, restricted to off-success thresholds and the full set).

Usage:
    python scripts/run_quickbind_posebusters.py --quickbind-dir <path/to/QuickBind clone>

Setup (see requirements/quickbind.txt for the full dependency story):
1. Clone aqlaboratory/QuickBind and the pinned aqlaboratory/openfold fork
   it depends on.
2. Download the PoseBusters Benchmark set (Zenodo record 8278563,
   posebusters_paper_data.zip) and symlink/copy its
   posebusters_benchmark_set/ directory into
   <quickbind-dir>/data/posebusters_benchmark_set/, with a `posebusters`
   names file listing the structure IDs to evaluate (one per line -
   posebusters_benchmark_set_ids.txt in that same Zenodo release, minus
   the one structure over 2000 residues that this repo's own evaluation
   excludes, per the paper).
3. QuickBind's own weights are bundled in the repo at
   checkpoints/quickbind_default/.
"""
import argparse
import sys
from pathlib import Path

import torch


def paired_t_stat(diffs):
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = (var / n) ** 0.5
    return mean / se if se > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quickbind-dir", required=True, type=Path)
    args = parser.parse_args()

    HERE = Path(__file__).parent
    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, str(args.quickbind_dir))

    from models.quickbind import load_model, correction_sites
    from correction.theory_c import install_correction_hooks

    assert torch.cuda.is_available(), "CUDA not available - refusing CPU fallback"
    device = torch.device("cuda")

    model, cfg = load_model(str(args.quickbind_dir))
    model = model.to(device)

    mode_flag = {"value": "off"}
    install_correction_hooks(correction_sites(model), q=0.9, mode_flag=mode_flag)

    from dataset.dataimporter import DataImporter

    test_data = DataImporter(
        complex_names_path=str(args.quickbind_dir / "data" / "posebusters_benchmark_set" / "posebusters"),
        **cfg["dataset_params"],
    )
    print(f"n structures: {len(test_data)}\n")

    one_hot_adj = cfg["model_parameters"].get("one_hot_adj", False)
    use_topological_distance = cfg["model_parameters"].get("use_topological_distance", False)

    def collate(data):
        aatype = data.aatype.unsqueeze(0).to(device)
        lig_atom_features = data.lig_atom_features.unsqueeze(0).to(dtype=torch.float32).to(device)
        t_true = data.true_lig_atom_coords.unsqueeze(0).to(device)
        rec_mask = torch.ones(data.aatype.shape[0]).unsqueeze(0).to(device)
        lig_mask = torch.ones(data.lig_atom_features.shape[0]).unsqueeze(0).to(device)
        if use_topological_distance:
            adj = torch.clamp(data.distance_matrix.unsqueeze(0), max=7).to(device)
            adj = torch.nn.functional.one_hot(adj.long(), num_classes=8).to(dtype=torch.float32)
        elif one_hot_adj:
            adj = data.adjacency_bo.unsqueeze(0).to(dtype=torch.int64).to(device)
        else:
            adj = data.adjacency.unsqueeze(0).unsqueeze(-1).to(dtype=torch.float32).to(device)
        ri = data.residue_index.unsqueeze(0).to(dtype=torch.int64).to(device)
        chain_id = data.chain_ids_processed.unsqueeze(0).to(dtype=torch.int64).to(device)
        entity_id = data.entity_ids_processed.unsqueeze(0).to(dtype=torch.int64).to(device)
        sym_id = data.sym_ids_processed.unsqueeze(0).to(dtype=torch.int64).to(device)
        id_batch = (ri, chain_id, entity_id, sym_id)
        t_rec = data.c_alpha_coords.unsqueeze(0).to(device)
        N = data.n_coords.unsqueeze(0).to(device)
        C = data.c_coords.unsqueeze(0).to(device)
        t_lig = data.lig_atom_coords.unsqueeze(0).to(device)
        pseudo_N = data.pseudo_N.unsqueeze(0).to(device)
        pseudo_C = data.pseudo_C.unsqueeze(0).to(device)
        return (
            aatype, lig_atom_features, adj, rec_mask, lig_mask, N, t_rec, C, t_lig, id_batch, pseudo_N, pseudo_C
        ), t_true, data.complex_name

    results = []
    n_failed = 0
    with torch.no_grad():
        for i in range(len(test_data)):
            try:
                batch, t_true, name = collate(test_data[i])
                _, _, _, rec_mask, _, _, _, _, _, _, _, _ = batch

                mode_flag["value"] = "off"
                pred_off = model(*batch)[-1][:, rec_mask.shape[-1]:].get_trans()[0]
                mode_flag["value"] = "on"
                pred_on = model(*batch)[-1][:, rec_mask.shape[-1]:].get_trans()[0]
                mode_flag["value"] = "off"

                truth = t_true[0]
                r_off = ((pred_off - truth) ** 2).sum(-1).mean().sqrt().item()
                r_on = ((pred_on - truth) ** 2).sum(-1).mean().sqrt().item()
            except Exception as e:
                print(f"  [{i}] FAILED: {e}", flush=True)
                n_failed += 1
                continue
            results.append((name, r_off, r_on))
            if len(results) % 20 == 0:
                print(f"  {len(results)}/{len(test_data)} done", flush=True)

    print(f"\nn evaluated: {len(results)}  (failed: {n_failed})\n")
    print(f"{'Off-success threshold':>22} {'n':>5} {'Uncorrected':>12} {'Corrected':>10} {'Rel. change':>12} {'Impr.':>10} {'t':>8}")
    for threshold in [2, 3, 5, None]:
        subset = [(o, c) for _, o, c in results if threshold is None or o < threshold]
        if not subset:
            continue
        n = len(subset)
        mean_off = sum(o for o, c in subset) / n
        mean_on = sum(c for o, c in subset) / n
        improved = sum(1 for o, c in subset if c < o)
        diffs = [c - o for o, c in subset]
        t_stat = paired_t_stat(diffs)
        label = "None (full set)" if threshold is None else f"<{threshold} A"
        rel_change = 100 * (mean_off - mean_on) / mean_off
        print(f"{label:>22} {n:>5} {mean_off:>12.4f} {mean_on:>10.4f} {rel_change:>+11.2f}% {improved:>5}/{n:<4} {t_stat:>8.2f}")


if __name__ == "__main__":
    main()
