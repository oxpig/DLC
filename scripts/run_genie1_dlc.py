"""Evaluates DLC correction on Genie1 (aqlaboratory/genie), the
predecessor to Genie3 (see genie3.py). Reproduces the same
noise-prediction-loss methodology used for Genie3's own DLC table: at 100
evenly spaced diffusion timesteps t (5 to 995, step 10), sample noise, run
the denoiser, and compare corrected vs uncorrected loss between the
model's implied predicted noise and the true injected noise - here on the
same nanobody-100 set used throughout the rest of the paper
(data/nanobody_100_pdb_codes.txt, built via data/prepare_splits.py, CA
coordinates only).

Only the undamped structure-net sites are corrected (genie1.
correction_sites - 10 for the published checkpoints, since these use
n_structure_layer=5, unlike Genie2/3's 8). The trunk's triangle-update
sites use a residual-bypass pattern and are not corrected.
"""
import argparse
from pathlib import Path

import torch

from correction.theory_c import install_correction_hooks
from models import genie1

Q_STRUCTURE = 0.9  # keep-prob for ipa_dropout / structure_transition_dropout (rate 0.1)


def load_ca_items(names, structures_dir, device):
    items = []
    for name in names:
        item = torch.load(Path(structures_dir) / f"{name}.pt", map_location="cpu", weights_only=False)
        ca = item["all_atom_positions"][:, 1, :].unsqueeze(0).to(device)  # (1, n, 3), atom37 index 1 = CA
        mask = item["seq_mask"].unsqueeze(0).float().to(device)
        items.append((name, ca, mask))
    return items


def build_frames(ca, mask, device):
    from genie.utils.affine_utils import T
    from genie.utils.geo_utils import compute_frenet_frames

    trans = ca - ca.mean(dim=1, keepdim=True)
    rots = compute_frenet_frames(trans, mask).to(device)
    return T(rots, trans)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, help="e.g. weights/scope_l_128 (must contain 'configuration' and one *.ckpt)")
    parser.add_argument("--structures-dir", required=True)
    parser.add_argument("--names-file", required=True, help="plain newline-separated structure name list, e.g. data/nanobody_100_pdb_codes.txt")
    parser.add_argument("--n-structures", type=int, default=4)
    parser.add_argument("--n-noise-draws", type=int, default=5)
    parser.add_argument("--n-t-values", type=int, default=100, help="evenly-spaced timesteps sampled from the 5..995 step-10 grid")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    (ckpt_path,) = list(ckpt_dir.glob("*.ckpt"))
    model = genie1.load_model(ckpt_dir / "configuration", ckpt_path, device=str(device))
    model.setup_schedule()
    model.setup = True

    mode_flag = {"value": "off"}
    install_correction_hooks(genie1.correction_sites(model), q=Q_STRUCTURE, mode_flag=mode_flag)

    with open(args.names_file) as f:
        names = [line.strip() for line in f if line.strip()][: args.n_structures]
    items = load_ca_items(names, args.structures_dir, device)

    full_t_grid = list(range(5, 1000, 10))  # 100 evenly spaced values, matching Genie3's table
    step = max(1, len(full_t_grid) // args.n_t_values)
    t_values = full_t_grid[::step][: args.n_t_values]

    torch.manual_seed(args.seed)
    losses_off, losses_on = [], []
    for name, ca, mask in items:
        name_off, name_on = [], []
        for draw in range(args.n_noise_draws):
            for t in t_values:
                t0 = build_frames(ca, mask, device)
                s = torch.tensor([t], device=device)
                ts, tnoise = model.q(t0, s, mask)  # sampled once, reused for both passes below

                mode_flag["value"] = "off"
                with torch.no_grad():
                    loss_off = model.loss_fn(tnoise, ts, s, mask).item()

                mode_flag["value"] = "on"
                with torch.no_grad():
                    loss_on = model.loss_fn(tnoise, ts, s, mask).item()
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
    print(f"\n=== genie1 ({ckpt_path.name}), n_structures={len(items)}, "
          f"n_noise_draws={args.n_noise_draws}, n_t={len(t_values)} ({n} total) ===")
    print(f"Uncorrected loss:    {mean_off:.4f}")
    print(f"+DLC-corrected loss: {mean_on:.4f}  ({100*(mean_off-mean_on)/mean_off:+.2f}%, {n_improved}/{n} improved)")


if __name__ == "__main__":
    main()
