"""Appendix figure: per-structure whole-domain RMSD shift (corrected minus
uncorrected) for ESMFold (no-recycle) and OpenFold (no-recycle, real MSA)
on the nanobody-100 set -- the same headline setting as Table~tab:nanobody
-- with a paired t-test per model (matching the paired-t convention already
used for the QuickBind table).

Usage:
    python scripts/plot_esmfold_openfold_shift.py --training-dir <path to FlashABB/training>

Reads already-saved raw structures (esmfold_nanobody_outputs/coords and
openfold_correction_outputs_recycle0/coords under --training-dir); does not
re-run inference.
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

BB37 = [0, 1, 2, 4]  # N, CA, C, O indices into atom37

TEAL = "#1b9e77"    # improved (Dark2)
ORANGE = "#d95f02"  # worsened (Dark2)


def extract_backbone(atom14):
    return atom14[:, :4, :]


def kabsch_Rt(x, y, mask):
    mask = mask.float().unsqueeze(-1)
    x, y = x * mask, y * mask
    n_valid = mask.sum()
    p_bar, q_bar = x.sum(0) / n_valid, y.sum(0) / n_valid
    x_c, y_c = x - p_bar, y - q_bar
    S = (x_c * mask).T @ (y_c * mask) / n_valid
    S = S + 1e-5 * torch.eye(3)
    U, _, Vh = torch.linalg.svd(S)
    V, Uh = Vh.T, U.T
    d = torch.det(V @ Uh)
    correction = torch.eye(3)
    correction[-1, -1] = d
    R = V @ correction @ Uh
    t = q_bar - R @ p_bar
    return R, t


def apply_Rt(points, R, t):
    return points @ R.T + t


def rmsd(a, b, mask):
    mask = mask.float()
    d2 = ((a - b) ** 2).sum(-1)
    return torch.sqrt((d2 * mask).sum() / mask.sum()).item()


def seq_whole_rmsd(pred, truth, mask):
    bb_pred, bb_truth = extract_backbone(pred), extract_backbone(truth)
    resolved = mask > 0
    valid = (bb_truth.abs().sum(-1) > 1e-6) & (bb_pred.abs().sum(-1) > 1e-6)
    bb_mask = (resolved.unsqueeze(-1) & valid).reshape(-1)
    R, t = kabsch_Rt(bb_truth.reshape(-1, 3), bb_pred.reshape(-1, 3), bb_mask)
    aligned = apply_Rt(bb_truth.reshape(-1, 3), R, t)
    return rmsd(aligned, bb_pred.reshape(-1, 3), bb_mask)


def esmfold_pairs(training_dir):
    rows = []
    for f in sorted(glob.glob(str(training_dir / "esmfold_nanobody_outputs/coords/*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        mask = d["truth_mask"]
        off = seq_whole_rmsd(d["off_atom14"], d["truth_atom14"], mask)
        on = seq_whole_rmsd(d["theory_atom14"], d["truth_atom14"], mask)
        rows.append((d["name"], off, on))
    return rows


def openfold_pairs(training_dir, data_dir):
    rows = []
    for f in sorted(glob.glob(str(training_dir / "openfold_correction_outputs_recycle0/coords/*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        name = d["name"]
        data = torch.load(data_dir / "structures" / "structures" / f"{name}.pt", map_location="cpu", weights_only=False)
        seq_mask = data["seq_mask"] * data["backbone_rigid_mask"]
        off_bb = d["off_atom37"][:, BB37, :]
        theory_bb = d["theory_atom37"][:, BB37, :]
        truth_bb = d["truth_atom37"][:, BB37, :]
        off = seq_whole_rmsd(off_bb, truth_bb, seq_mask)
        on = seq_whole_rmsd(theory_bb, truth_bb, seq_mask)
        rows.append((name, off, on))
    return rows


def plot_panel(ax, off, on, title):
    off, on = np.array(off), np.array(on)
    delta = on - off
    t_stat, p_val = stats.ttest_rel(on, off)
    improved = delta < 0
    n = len(delta)

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.32, 0.32, size=n)
    y = jitter
    colors = np.where(improved, TEAL, ORANGE)
    ax.scatter(delta, y, c=colors, s=22, alpha=0.75, linewidths=0, zorder=3)
    ax.axvline(0.0, color="0.3", linewidth=1.2, zorder=2)
    ax.axvline(delta.mean(), color="0.15", linewidth=1.6, linestyle="--", zorder=4)

    mean_off, mean_on = off.mean(), on.mean()
    pct = 100 * (mean_off - mean_on) / mean_off
    p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
    ax.set_title(
        f"{title}\nmean {mean_off:.3f}→{mean_on:.3f}Å ({pct:+.2f}%), "
        f"{improved.sum()}/{n} improved\n$t$={t_stat:.2f}, {p_str}",
        fontsize=10,
    )
    ax.set_xlabel("Corrected − uncorrected whole-domain RMSD (Å)")
    ax.set_yticks([])
    ax.set_ylim(-0.6, 0.6)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    return t_stat, p_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path,
                         help="Path to flashabb_nanobody_training/data (for OpenFold's seq_mask)")
    parser.add_argument("--out", default="figures/fig_esmfold_openfold_shift.png")
    args = parser.parse_args()

    esm_rows = esmfold_pairs(args.training_dir)
    of_rows = openfold_pairs(args.training_dir, args.data_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    plot_panel(axes[0], [r[1] for r in esm_rows], [r[2] for r in esm_rows], "ESMFold (no recycling)")
    plot_panel(axes[1], [r[1] for r in of_rows], [r[2] for r in of_rows], "OpenFold (no recycling, real MSA)")

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL, markersize=7, label="Improved"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markersize=7, label="Worsened"),
        plt.Line2D([0], [0], color="0.15", linestyle="--", linewidth=1.6, label="Mean shift"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
