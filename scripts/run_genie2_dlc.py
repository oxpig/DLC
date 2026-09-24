"""Evaluates DLC correction on Genie2 (aqlaboratory/genie2), following
the same methodology as scripts/run_genie1_dlc.py (see its docstring) -
100 evenly spaced diffusion timesteps, corrected vs uncorrected
noise-prediction loss, on the same nanobody-100 set used throughout the
rest of the paper. Only the undamped structure-net sites are corrected
(16, since Genie2 uses n_structure_layer=8 like Genie3, unlike Genie1's
5). The trunk's triangle-update sites use a residual-bypass pattern and
are not corrected.

Genie2's own package is also top-level-named "genie" (same as Genie1's) -
this script assumes its repo root is first on PYTHONPATH so `import
genie` resolves to Genie2, not whichever venv has Genie1 pip-installed
(see models/genie2.py's module docstring). Also unlike Genie1, Genie2's
denoiser takes a full feature dictionary (motif-conditioning fields, all
inert here since fixed_sequence_mask is all-False - fully unconditional,
matching Genie1's own setup) rather than a bare (coords, mask) pair, and
its own diffusion schedule is 1-indexed (valid timesteps 1..n_timestep,
index 0 reserved for the un-noised case - see diffusion/schedule.py's
get_betas), unlike Genie1's 0-indexed schedule.
"""
import argparse
from pathlib import Path

import torch

from correction.theory_c import install_correction_hooks
from models import genie2

Q_STRUCTURE = 0.9  # keep-prob for ipa_dropout / structure_transition_dropout (rate 0.1)


def build_features(ca, device):
    """ca: (n, 3) CA coordinates for one single-chain structure. Builds a
    batch-of-1, fully unconditional (no motif) feature dict - matching
    genie.utils.feat_utils.create_empty_np_features's fields, but with
    real (not zero) atom_positions."""
    n = ca.shape[0]
    return {
        "num_chains": torch.tensor([1], device=device),
        "num_residues": torch.tensor([n], device=device),
        "num_residues_per_chain": torch.tensor([[n]], device=device),
        "aatype": torch.zeros((1, n, 20), device=device),
        "atom_positions": ca.unsqueeze(0).to(device),
        "residue_mask": torch.ones((1, n), device=device),
        "residue_index": torch.arange(n, device=device).unsqueeze(0),
        "chain_index": torch.zeros((1, n), device=device),
        "fixed_sequence_mask": torch.zeros((1, n), dtype=torch.bool, device=device),
        "fixed_structure_mask": torch.zeros((1, n, n), dtype=torch.bool, device=device),
        "fixed_group": torch.zeros((1, n), device=device),
        "interface_mask": torch.zeros((1, n), dtype=torch.bool, device=device),
    }


def noise_prediction_loss(model, features, t, device):
    """Replicates Genie's own training_step (diffusion/genie.py) for a
    single fixed timestep t instead of a random one, and without the
    motif condition/infill loss split (features['fixed_sequence_mask'] is
    all-False here, so that split is a no-op - this is plain masked MSE
    over every residue, matching Genie1's fully-unconditional test)."""
    from genie.utils.affine_utils import T
    from genie.utils.geo_utils import compute_frenet_frames
    from genie.utils.loss import mse

    mask = features["residue_mask"]
    s = torch.tensor([t], device=device)
    z = torch.randn_like(features["atom_positions"]) * mask.unsqueeze(-1)
    trans_s = model.sqrt_alphas_cumprod[s].view(-1, 1, 1) * features["atom_positions"] + \
        model.sqrt_one_minus_alphas_cumprod[s].view(-1, 1, 1) * z
    rots_s = compute_frenet_frames(trans_s, features["chain_index"], mask)
    ts = T(rots_s, trans_s)

    with torch.no_grad():
        output = model.model(ts, s, features)
    return mse(output["z"], z, mask, aggregate="mean").item()


def load_ca_items(names, structures_dir, device):
    items = []
    for name in names:
        item = torch.load(Path(structures_dir) / f"{name}.pt", map_location="cpu", weights_only=False)
        ca = item["all_atom_positions"][:, 1, :]  # (n, 3), atom37 index 1 = CA
        ca = ca - ca.mean(dim=0, keepdim=True)
        items.append((name, ca))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to epoch=*.ckpt")
    parser.add_argument("--config", required=True, help="path to the 'configuration' file alongside it")
    parser.add_argument("--structures-dir", required=True)
    parser.add_argument("--names-file", required=True, help="plain newline-separated structure name list, e.g. data/nanobody_100_pdb_codes.txt")
    parser.add_argument("--n-structures", type=int, default=4)
    parser.add_argument("--n-noise-draws", type=int, default=5)
    parser.add_argument("--n-t-values", type=int, default=100, help="evenly-spaced timesteps sampled from the 5..995 step-10 grid")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = genie2.load_model(args.config, args.checkpoint, device=str(device))
    model.setup_schedule()
    model.setup = True

    mode_flag = {"value": "off"}
    install_correction_hooks(genie2.correction_sites(model), q=Q_STRUCTURE, mode_flag=mode_flag)

    with open(args.names_file) as f:
        names = [line.strip() for line in f if line.strip()][: args.n_structures]
    items = load_ca_items(names, args.structures_dir, device)

    full_t_grid = list(range(5, 1000, 10))  # 100 evenly spaced values, matching Genie1/Genie3's table
    step = max(1, len(full_t_grid) // args.n_t_values)
    t_values = full_t_grid[::step][: args.n_t_values]

    torch.manual_seed(args.seed)
    losses_off, losses_on = [], []
    for name, ca in items:
        features = build_features(ca, device)
        name_off, name_on = [], []
        for draw in range(args.n_noise_draws):
            for t in t_values:
                rng_state = torch.get_rng_state()

                mode_flag["value"] = "off"
                loss_off = noise_prediction_loss(model, features, t, device)

                torch.set_rng_state(rng_state)  # reuse the exact same noise draw for the "on" pass
                mode_flag["value"] = "on"
                loss_on = noise_prediction_loss(model, features, t, device)
                mode_flag["value"] = "off"

                name_off.append(loss_off)
                name_on.append(loss_on)
        losses_off.extend(name_off)
        losses_on.extend(name_on)
        print(f"  [{name}] mean off={sum(name_off)/len(name_off):.4f} on={sum(name_on)/len(name_on):.4f}", flush=True)

    n = len(losses_off)
    mean_off = sum(losses_off) / n
    mean_on = sum(losses_on) / n
    n_improved = sum(1 for o, c in zip(losses_off, losses_on) if c < o)
    print(f"\n=== genie2 ({Path(args.checkpoint).name}), n_structures={len(items)}, "
          f"n_noise_draws={args.n_noise_draws}, n_t={len(t_values)} ({n} total) ===")
    print(f"Uncorrected loss:    {mean_off:.4f}")
    print(f"+DLC-corrected loss: {mean_on:.4f}  ({100*(mean_off-mean_on)/mean_off:+.2f}%, {n_improved}/{n} improved)")


if __name__ == "__main__":
    main()
