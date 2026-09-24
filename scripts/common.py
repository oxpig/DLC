"""Shared evaluation helpers used by scripts/run_nanobody_table.py and
scripts/run_headline_table.py. Each function evaluates one model on a list
of prepared structures (from data/prepare_splits.py), off vs corrected,
and returns a list of (name, rmsd_off, rmsd_on) tuples - the structures
passed in can be single-chain (nanobody) or paired heavy+light, both work
transparently since chain identity comes from each item's own is_heavy
field, not from which script is calling."""
from pathlib import Path

import torch

from data.features import pair_features_paired
from data.metrics import seq_whole_rmsd

DROPOUT_RATE = 0.1
Q = 1.0 - DROPOUT_RATE


def load_structures(names_file, structures_dir):
    """Nanobody names in nanobody_100_pdb_codes.txt are already full
    instance names (pdb_<PDB_ID>_<Hchain>_+.pt). Headline names in
    headline_34_pdb_codes.txt are bare 4-char PDB codes - prepare_splits.py
    saves those under their resolved INSTANCE name instead
    (pdb_0000<code>_<Hchain>_<Lchain>.pt), so fall back to a glob match."""
    with open(names_file) as f:
        names = [line.strip() for line in f if line.strip()]
    items = []
    for name in names:
        path = Path(structures_dir) / f"{name}.pt"
        if not path.is_file():
            matches = list(Path(structures_dir).glob(f"pdb_0000{name}_*.pt"))
            if not matches:
                print(f"  [{name}] missing, skipping")
                continue
            path = matches[0]
        items.append((name, torch.load(path, map_location="cpu", weights_only=False)))
    return items


def eval_pair_rep_model(model_name, model, mode_flag, items, device, plm_dir=None):
    """ABB3/Ibex/NbForge/FlashABB share this input convention: single =
    one-hot aatype (+ one-hot chain, + one-hot apo-flag for Ibex). ABB3/
    Ibex/NbForge additionally need an explicit pair representation
    (relative-position + chain features from pair_features_paired);
    FlashABB computes relative-position info internally from res_idx
    (passed as a required positional argument instead) and takes no
    separate pair input."""
    results = []
    for name, item in items:
        aatype = item["aatype"].unsqueeze(0).to(device)
        res_idx = item["residue_index"].to(device)
        seq_mask = item["seq_mask"].unsqueeze(0).to(device)
        is_heavy = item["is_heavy"].to(device)
        single_aa = torch.nn.functional.one_hot(aatype, 21).float()

        single_chain = torch.nn.functional.one_hot(is_heavy.long(), 2).unsqueeze(0).float()
        if model_name == "ibex":
            pair = pair_features_paired(res_idx, is_heavy).unsqueeze(0)
            is_apo = torch.zeros_like(aatype)
            single_conf = torch.nn.functional.one_hot(is_apo, 2).float()
            single = torch.cat((single_aa, single_chain, single_conf), dim=-1)
            plm_embedding = torch.load(f"{plm_dir}/{name}.pt", map_location=device, weights_only=False).unsqueeze(0)
            forward = lambda: model({"single": single, "pair": pair}, aatype.clone(), plm_embedding, mask=seq_mask)
        elif model_name == "flashabb":
            single = torch.cat((single_aa, single_chain), dim=-1)
            forward = lambda: model({"single": single}, aatype.clone(), res_idx.unsqueeze(0), mask=seq_mask)
        else:
            pair = pair_features_paired(res_idx, is_heavy).unsqueeze(0)
            single = torch.cat((single_aa, single_chain), dim=-1)
            forward = lambda: model({"single": single, "pair": pair}, aatype.clone(), mask=seq_mask)

        truth_atom14 = item["atom14_gt_positions"].to(device)
        whole_seq_mask = item["seq_mask"].to(device)

        with torch.no_grad():
            mode_flag["value"] = "off"
            atom14_off = forward()["positions"][-1, 0]
            mode_flag["value"] = "on"
            atom14_on = forward()["positions"][-1, 0]
        mode_flag["value"] = "off"

        r_off = seq_whole_rmsd(atom14_off, truth_atom14, whole_seq_mask)
        r_on = seq_whole_rmsd(atom14_on, truth_atom14, whole_seq_mask)
        results.append((name, r_off, r_on))
        print(f"  [{name}] off={r_off:.4f} on={r_on:.4f}", flush=True)
    return results


AA_1LETTER = "ACDEFGHIKLMNPQRSTVWYX"


def _sequence_for_esmfold(item):
    """ESMFold's multimer convention: chains joined by ':' (its own
    internal glycine-linker + residue-index-offset handling takes care of
    the rest). Single-chain (nanobody) structures just pass their own
    sequence through unchanged."""
    is_heavy = item["is_heavy"]
    if bool((is_heavy == 0).any()):
        heavy_seq = item["sequence"][: int(is_heavy.sum().item())]
        light_seq = item["sequence"][int(is_heavy.sum().item()):]
        return f"{heavy_seq}:{light_seq}"
    return item["sequence"]


def eval_esmfold(model, mode_flag, items, device):
    """ESMFold's own `positions` output is already atom14-shaped (N/CA/C/O
    in the first four slots, bit-exact bond lengths verified against real
    physical values) - same convention as our own truth data, no
    reindexing needed."""
    results = []
    for name, item in items:
        sequence = _sequence_for_esmfold(item)
        truth_atom14 = item["atom14_gt_positions"].to(device)
        seq_mask = item["seq_mask"].to(device)

        def run(mode):
            mode_flag["value"] = mode
            with torch.no_grad():
                out = model.infer(sequence, num_recycles=0)
            mode_flag["value"] = "off"
            atom14 = out["positions"][-1, 0]
            assert atom14.shape[0] == truth_atom14.shape[0], (
                f"{atom14.shape[0]} vs {truth_atom14.shape[0]} - ESMFold's own chain-linker "
                f"residues must be stripped for paired input, not yet handled here"
            )
            return seq_whole_rmsd(atom14, truth_atom14, seq_mask)

        r_off, r_on = run("off"), run("on")
        results.append((name, r_off, r_on))
        print(f"  [{name}] off={r_off:.4f} on={r_on:.4f}", flush=True)
    return results


def _atom37_backbone_to_atom14(atom37):
    """OpenFold's `final_atom_positions` is atom37-shaped, using the global
    atom_types order N/CA/C/CB/O/... (CB before O) - NOT our atom14
    convention's N/CA/C/O first-four-slots order. Pulls out just the real
    backbone atoms (indices 0,1,2,4 in atom37) into the atom14 layout
    seq_whole_rmsd expects, so `extract_backbone`'s [:, :4, :] slice is
    correct."""
    n, ca, c, o = atom37[:, 0, :], atom37[:, 1, :], atom37[:, 2, :], atom37[:, 4, :]
    return torch.stack([n, ca, c, o], dim=1)


def eval_openfold(model, cfg, mode_flag, items, device, msa_dir):
    """Single-chain only (nanobody set) - real per-chain MSAs are fetched
    per sequence, and OpenFold's own monomer feature pipeline (used here)
    doesn't handle paired multi-chain input. A paired headline-table row
    would need the AlphaFold-Gap trick (concatenated sequence, residue-index
    offset at the chain break, block-diagonal MSA) which isn't implemented
    here - raises NotImplementedError for paired items rather than silently
    running the model on a paired sequence as if it were a monomer."""
    from models.openfold import build_batch

    results = []
    for name, item in items:
        if bool((item["is_heavy"] == 0).any()):
            raise NotImplementedError(
                f"[{name}] OpenFold paired-chain (headline table) input needs the "
                "AlphaFold-Gap trick, not implemented in this repo - see module docstring"
            )
        a3m_path = f"{msa_dir}/{name}.a3m"
        if not Path(a3m_path).is_file():
            print(f"  [{name}] missing MSA, skipping")
            continue
        sequence = item["sequence"]
        batch = build_batch(sequence, a3m_path, cfg, device=device)
        truth_atom14 = item["atom14_gt_positions"].to(device)
        seq_mask = item["seq_mask"].to(device)

        def run(mode):
            mode_flag["value"] = mode
            with torch.no_grad():
                out = model(batch)
            mode_flag["value"] = "off"
            atom14 = _atom37_backbone_to_atom14(out["final_atom_positions"][0])
            return seq_whole_rmsd(atom14, truth_atom14, seq_mask)

        r_off, r_on = run("off"), run("on")
        results.append((name, r_off, r_on))
        print(f"  [{name}] off={r_off:.4f} on={r_on:.4f}", flush=True)
    return results


def _immunebuilder_atom14(all_atoms):
    """ImmuneBuilder's `StructureModule.forward` returns atoms in its OWN
    per-residue order (`ImmuneBuilder.constants.rel_pos`/`residue_atoms`:
    CA, N, C, CB, O, ... - CA is the backbone-frame origin, listed first)
    - NOT OpenFold's atom37 order (N, CA, C, CB, O, ...). Confirmed
    directly from `ImmuneBuilder.constants.rel_pos['A']`: index 0 is the
    local-frame origin (0,0,0) = CA; index 1 is ~1.46A away = N. Using
    `_atom37_backbone_to_atom14` (built for OpenFold's order) on this output
    silently swaps N and CA in every downstream RMSD - this is the correct
    indexing for ImmuneBuilder specifically."""
    ca, n, c, o = all_atoms[:, 0, :], all_atoms[:, 1, :], all_atoms[:, 2, :], all_atoms[:, 4, :]
    return torch.stack([n, ca, c, o], dim=1)


def eval_immunebuilder(model_name, models, items, device):
    """ABodyBuilder2/NanoBodyBuilder2 (public `ImmuneBuilder` package) share
    the same architecture, which instantiates its dropout+LayerNorm sites
    with dropout=0.0 architecturally (never overridden) - so there is no
    correction to apply here, off and on are always identical by
    construction (see the model modules' own docstrings). Reports plain
    accuracy only.

    `models`: dict of {member_name: model}, e.g. from load_ensemble() - runs
    every member and reports both (a) the first-listed member alone and (b)
    the full ensemble under ImmuneBuilder's own blind (no-ground-truth)
    selection rule: align every member's CA trace onto a common frame (via
    ImmuneBuilder.util.find_alignment_transform, the exact function
    ImmuneBuilder's own Antibody.ranking uses), then pick the member closest
    to the cross-model consensus mean - not just whichever was listed first.

    Reports both whole-domain and framework-aligned CDR-H3 RMSD (region
    codes from data/prepare_splits.py's region_numeric: 6-9=heavy framework,
    2=heavy CDR3 - unambiguous vs. light-chain codes by construction, no
    extra chain masking needed).

    Returns (name, r_off, r_on) whole-domain tuples for the FULL ENSEMBLE's
    pick (matching the other eval_* functions' shape, off==on since
    correction is a no-op) - CDR-H3 and the single-member baseline are
    printed directly since they don't fit that 3-tuple shape."""
    from ImmuneBuilder.util import get_encoding, find_alignment_transform

    from data.metrics import cdrh3_fw_aligned_rmsd

    FW_CODES = torch.tensor([6, 7, 8, 9], device=device)
    CDR3_CODE = torch.tensor([2], device=device)
    member_names = list(models)
    first_member = member_names[0]

    results = []
    member1_whole, member1_cdrh3, ensemble_whole, ensemble_cdrh3 = [], [], [], []
    for name, item in items:
        is_heavy = item["is_heavy"]
        seq = item["sequence"]
        if bool((is_heavy == 0).any()):
            n_heavy = int(is_heavy.sum().item())
            heavy_seq, light_seq = seq[:n_heavy], seq[n_heavy:]
            encoding = torch.tensor(get_encoding({"H": heavy_seq, "L": light_seq}), dtype=torch.get_default_dtype(), device=device)
            full_seq = heavy_seq + light_seq
        else:
            encoding = torch.tensor(get_encoding({"H": seq}, "H"), dtype=torch.get_default_dtype(), device=device)
            full_seq = seq

        truth_atom14 = item["atom14_gt_positions"].to(device)
        seq_mask = item["seq_mask"].to(device)
        region_numeric = item["region_numeric"].to(device)
        fw_mask = torch.isin(region_numeric, FW_CODES)
        cdr3_mask = torch.isin(region_numeric, CDR3_CODE)

        atom14_by_member = {}
        with torch.no_grad():
            for member_name, model in models.items():
                all_atoms, _ = model(encoding, full_seq)
                atom14_by_member[member_name] = _immunebuilder_atom14(all_atoms)

        r1_whole = seq_whole_rmsd(atom14_by_member[first_member], truth_atom14, seq_mask)
        r1_cdrh3 = cdrh3_fw_aligned_rmsd(atom14_by_member[first_member], truth_atom14, seq_mask, fw_mask, cdr3_mask)
        member1_whole.append(r1_whole)
        member1_cdrh3.append(r1_cdrh3)

        ca_traces = torch.stack([atom14_by_member[m][:, 1, :] for m in member_names])  # (M, N, 3)
        R, t = find_alignment_transform(ca_traces)
        aligned = (ca_traces - t) @ R
        error_estimates = (aligned - aligned.mean(0, keepdim=True)).square().sum(-1)
        best_member = member_names[error_estimates.mean(-1).argmin().item()]

        e_whole = seq_whole_rmsd(atom14_by_member[best_member], truth_atom14, seq_mask)
        e_cdrh3 = cdrh3_fw_aligned_rmsd(atom14_by_member[best_member], truth_atom14, seq_mask, fw_mask, cdr3_mask)
        ensemble_whole.append(e_whole)
        ensemble_cdrh3.append(e_cdrh3)
        results.append((name, e_whole, e_whole))

        print(f"  [{name}] {first_member}: whole={r1_whole:.4f} cdrh3={r1_cdrh3:.4f}   "
              f"ensemble({best_member}): whole={e_whole:.4f} cdrh3={e_cdrh3:.4f}", flush=True)

    n = len(results)
    print(f"\n=== {model_name}, {first_member} only (n={n}) ===")
    print(f"whole-domain: {sum(member1_whole)/n:.4f}   CDR-H3: {sum(member1_cdrh3)/n:.4f}")
    print(f"\n=== {model_name}, {len(member_names)}-member ensemble, blind consensus ranking (n={n}) ===")
    print(f"whole-domain: {sum(ensemble_whole)/n:.4f}   CDR-H3: {sum(ensemble_cdrh3)/n:.4f}")
    return results


def run_model(model_name, checkpoint, items, device, plm_embeddings_dir=None, msa_dir=None):
    mode_flag = {"value": "off"}

    if model_name == "abb3":
        from models.abb3 import load_model, correction_sites
        model = load_model(checkpoint).to(device)
        from correction.theory_c import install_correction_hooks
        install_correction_hooks(correction_sites(model), q=Q, mode_flag=mode_flag)
        return eval_pair_rep_model(model_name, model, mode_flag, items, device)

    if model_name == "ibex":
        from models.ibex import load_model, correction_sites, plm_correction_site
        from correction.theory_c import install_correction_hooks
        assert plm_embeddings_dir, "--plm-embeddings-dir is required for ibex"
        model = load_model(checkpoint).to(device)
        install_correction_hooks(correction_sites(model), q=Q, mode_flag=mode_flag)
        plm_sites, plm_q = plm_correction_site(model)
        install_correction_hooks(plm_sites, q=plm_q, mode_flag=mode_flag)
        return eval_pair_rep_model(model_name, model, mode_flag, items, device, plm_dir=plm_embeddings_dir)

    if model_name == "nbforge":
        from models.nbforge import load_model, correction_sites, RECOMMENDED_Q
        from correction.theory_c import install_correction_hooks
        model = load_model(checkpoint).to(device)
        install_correction_hooks(correction_sites(model), q=RECOMMENDED_Q, mode_flag=mode_flag)
        return eval_pair_rep_model(model_name, model, mode_flag, items, device)

    if model_name == "flashabb":
        from models.flashabb import load_model, correction_sites
        from correction.theory_c import install_correction_hooks
        model = load_model(device=str(device))
        install_correction_hooks(correction_sites(model), q=Q, mode_flag=mode_flag)
        return eval_pair_rep_model(model_name, model, mode_flag, items, device)

    if model_name == "esmfold":
        from models.esmfold import load_model, correction_sites
        from correction.theory_c import install_correction_hooks
        model = load_model().to(device)
        install_correction_hooks(correction_sites(model), q=Q, mode_flag=mode_flag)
        return eval_esmfold(model, mode_flag, items, device)

    if model_name == "openfold":
        from models.openfold import load_model, correction_sites
        from correction.theory_c import install_correction_hooks
        assert msa_dir, "--msa-dir is required for openfold"
        model, cfg = load_model(checkpoint, num_recycles=0)
        model = model.to(device)
        install_correction_hooks(correction_sites(model), q=Q, mode_flag=mode_flag)
        return eval_openfold(model, cfg, mode_flag, items, device, msa_dir)

    if model_name == "abb2":
        from models.abb2 import load_ensemble
        assert checkpoint, "--checkpoint (a local weights cache directory, auto-downloaded from Zenodo) is required for abb2"
        models = {name: m.to(device) for name, m in load_ensemble(checkpoint).items()}
        return eval_immunebuilder(model_name, models, items, device)

    if model_name == "nanobodybuilder2":
        from models.nanobodybuilder2 import load_ensemble
        assert checkpoint, "--checkpoint (a local weights cache directory, auto-downloaded from Zenodo) is required for nanobodybuilder2"
        models = {name: m.to(device) for name, m in load_ensemble(checkpoint).items()}
        return eval_immunebuilder(model_name, models, items, device)

    raise ValueError(f"unknown model {model_name}")


def print_summary(model_name, set_name, results):
    n = len(results)
    mean_off = sum(r[1] for r in results) / n
    mean_on = sum(r[2] for r in results) / n
    improved = sum(1 for r in results if r[2] < r[1])
    print(f"\n=== {model_name}, {set_name} (n={n}) ===")
    print(f"whole-structure RMSD off: {mean_off:.4f}")
    print(f"whole-structure RMSD on:  {mean_on:.4f}  ({100*(mean_off-mean_on)/mean_off:+.2f}%, {improved}/{n} improved)")
