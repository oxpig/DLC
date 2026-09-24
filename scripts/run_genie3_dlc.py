"""Evaluates DLC correction on Genie3 (aqlaboratory/genie3), following the
same methodology as scripts/run_genie1_dlc.py / run_genie2_dlc.py (see
their docstrings) - evenly spaced diffusion timesteps, corrected vs
uncorrected noise-prediction loss, on the nanobody-100 set used throughout
the paper (data/nanobody_100_pdb_codes.txt, built via data/prepare_splits.py,
CA coordinates only).

All 26 undamped sites are corrected (models/genie3.py's correction_sites:
structure_net's 8 blocks x 2 sites plus pair_transform_net's 5 blocks x 2
sites). The trunk's triangle-update sites use a residual-bypass pattern and
are not corrected.

Genie3's own top-level package is "genie3", not "genie" - no PYTHONPATH
shadowing needed here (unlike genie2.py), assuming the genie3 package is
on the path as it already is via genie3_venv's editable install.
"""
import argparse
from pathlib import Path

import torch

from correction.theory_c import install_correction_hooks
from models import genie3

Q = 0.9  # keep-prob for ipa_dropout / structure_transition_dropout (rate 0.1), all 26 sites


def load_ca_items(names, structures_dir):
    items = []
    for name in names:
        item = torch.load(Path(structures_dir) / f"{name}.pt", map_location="cpu", weights_only=False)
        ca = item["all_atom_positions"][:, 1, :]  # (n, 3), atom37 index 1 = CA
        ca = ca - ca.mean(dim=0, keepdim=True)
        items.append((name, ca))
    return items


def build_batch(ca, device):
    from genie3.generation.utils.feat_utils import create_np_features_from_length

    n = ca.shape[0]
    np_features = create_np_features_from_length(n)
    np_features["gt_atom_positions"] = ca.numpy()

    batch = {}
    for k, v in np_features.items():
        t = torch.as_tensor(v)
        if k == "cond_group":
            t = t.long()
        elif t.dtype == torch.float64:
            t = t.float()
        elif t.dtype == torch.bool:
            t = t.float()
        batch[k] = t.unsqueeze(0).to(device)
    return batch


def noise_prediction_loss(diffusion, model, batch, t, device):
    from genie3.generation.utils.loss.mse import mse

    s = torch.tensor([t], device=device)
    with torch.no_grad():
        out = diffusion.training_step(model, batch, t=s)
    return mse(out["zl_out"], out["zl"], batch["gt_atom_mask"], aggregate="mean").item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to step=*.ckpt")
    parser.add_argument("--structures-dir", required=True)
    parser.add_argument("--names-file", required=True, help="plain newline-separated structure name list, e.g. data/nanobody_100_pdb_codes.txt")
    parser.add_argument("--n-structures", type=int, default=4)
    parser.add_argument("--n-noise-draws", type=int, default=5)
    parser.add_argument("--n-t-values", type=int, default=100, help="evenly-spaced timesteps sampled from the 5..995 step-10 grid")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = genie3.load_model(args.checkpoint).to(device)
    diffusion = genie3.load_diffusion(device=str(device))

    mode_flag = {"value": "off"}
    install_correction_hooks(genie3.correction_sites(model), q=Q, mode_flag=mode_flag)

    with open(args.names_file) as f:
        names = [line.strip() for line in f if line.strip()][: args.n_structures]
    items = load_ca_items(names, args.structures_dir)

    full_t_grid = list(range(5, 1000, 10))  # 100 evenly spaced values, matching Genie1/2/3's table
    step = max(1, len(full_t_grid) // args.n_t_values)
    t_values = full_t_grid[::step][: args.n_t_values]

    torch.manual_seed(args.seed)
    losses_off, losses_on = [], []
    for name, ca in items:
        batch = build_batch(ca, device)
        name_off, name_on = [], []
        for draw in range(args.n_noise_draws):
            for t in t_values:
                rng_state = torch.get_rng_state()

                mode_flag["value"] = "off"
                loss_off = noise_prediction_loss(diffusion, model, batch, t, device)

                torch.set_rng_state(rng_state)  # reuse the exact same noise draw for the "on" pass
                mode_flag["value"] = "on"
                loss_on = noise_prediction_loss(diffusion, model, batch, t, device)
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
    print(f"\n=== genie3 ({Path(args.checkpoint).name}), n_structures={len(items)}, "
          f"n_noise_draws={args.n_noise_draws}, n_t={len(t_values)} ({n} total) ===")
    print(f"Uncorrected loss:    {mean_off:.4f}")
    print(f"+DLC-corrected loss: {mean_on:.4f}  ({100*(mean_off-mean_on)/mean_off:+.2f}%, {n_improved}/{n} improved)")


if __name__ == "__main__":
    main()
