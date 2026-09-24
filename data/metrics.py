"""RMSD metrics shared across all model evaluation scripts: whole-structure
(single global alignment) and framework-aligned CDR-H3 (align on framework
backbone only, then measure CDR-H3 backbone RMSD), both operating on
atom14-shaped (N/CA/C/O in the first four slots) backbone coordinates.
"""
import torch


def kabsch_Rt(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    """x, y: (L, 3). mask: (L,) bool/float. Returns the rotation R (3,3) and
    translation t (3,) that best aligns x onto y over the masked points."""
    mask = mask.float().unsqueeze(-1)
    x = x * mask
    y = y * mask
    n_valid = mask.sum()
    p_bar = x.sum(dim=0) / n_valid
    q_bar = y.sum(dim=0) / n_valid
    x_c = x - p_bar
    y_c = y - q_bar
    S = (x_c * mask).T @ (y_c * mask) / n_valid
    S = S + 1e-5 * torch.eye(3, device=S.device)
    U, _, Vh = torch.linalg.svd(S)
    V = Vh.T
    Uh = U.T
    d = torch.det((V @ Uh).cpu()).to(V.device)
    correction = torch.eye(3, device=x.device)
    correction[-1, -1] = d
    R = V @ correction @ Uh
    t = q_bar - R @ p_bar
    return R, t


def apply_Rt(points: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return points @ R.T + t


def rmsd(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.float()
    d2 = ((a - b) ** 2).sum(dim=-1)
    return torch.sqrt((d2 * mask).sum() / mask.sum()).item()


def extract_backbone(atom14: torch.Tensor) -> torch.Tensor:
    """(L, 14, 3) -> (L, 4, 3): N, CA, C, O."""
    return atom14[:, :4, :]


def seq_whole_rmsd(atom14_pred: torch.Tensor, atom14_truth: torch.Tensor,
                    seq_mask: torch.Tensor, use_atom_validity_mask: bool = True) -> float:
    """Whole-structure backbone RMSD: single Kabsch alignment over all
    resolved backbone atoms."""
    bb_pred = extract_backbone(atom14_pred)
    bb_truth = extract_backbone(atom14_truth)

    resolved_mask = seq_mask > 0
    if use_atom_validity_mask:
        atom_valid = (bb_truth.abs().sum(dim=-1) > 1e-6) & (bb_pred.abs().sum(dim=-1) > 1e-6)
        bb_mask = (resolved_mask.unsqueeze(-1) & atom_valid).reshape(-1)
    else:
        bb_mask = resolved_mask.unsqueeze(-1).expand(-1, 4).reshape(-1)

    bb_pred_flat = bb_pred.reshape(-1, 3)
    bb_truth_flat = bb_truth.reshape(-1, 3)

    R, t = kabsch_Rt(bb_truth_flat, bb_pred_flat, bb_mask)
    truth_aligned = apply_Rt(bb_truth_flat, R, t)
    return rmsd(truth_aligned, bb_pred_flat, bb_mask)


def cdrh3_fw_aligned_rmsd(atom14_pred: torch.Tensor, atom14_truth: torch.Tensor,
                           seq_mask: torch.Tensor, framework_mask: torch.Tensor,
                           cdr3_mask: torch.Tensor, use_atom_validity_mask: bool = True) -> float:
    """Framework-aligned CDR-H3 backbone RMSD: Kabsch-align on framework
    backbone atoms only, then measure RMSD on CDR-H3 backbone atoms.
    framework_mask/cdr3_mask: (L,) bool, from an AHO-scheme region
    annotation (see data/prepare_splits.py)."""
    bb_pred = extract_backbone(atom14_pred)
    bb_truth = extract_backbone(atom14_truth)

    fw_mask = framework_mask & (seq_mask > 0)
    cdr3_mask = cdr3_mask & (seq_mask > 0)

    if use_atom_validity_mask:
        atom_valid = (bb_truth.abs().sum(dim=-1) > 1e-6) & (bb_pred.abs().sum(dim=-1) > 1e-6)
        fw_bb_mask = (fw_mask.unsqueeze(-1) & atom_valid).reshape(-1)
        cdr3_bb_mask = (cdr3_mask.unsqueeze(-1) & atom_valid).reshape(-1)
    else:
        fw_bb_mask = fw_mask.unsqueeze(-1).expand(-1, 4).reshape(-1)
        cdr3_bb_mask = cdr3_mask.unsqueeze(-1).expand(-1, 4).reshape(-1)

    bb_pred_flat = bb_pred.reshape(-1, 3)
    bb_truth_flat = bb_truth.reshape(-1, 3)

    R, t = kabsch_Rt(bb_truth_flat, bb_pred_flat, fw_bb_mask)
    truth_aligned = apply_Rt(bb_truth_flat, R, t)
    return rmsd(truth_aligned, bb_pred_flat, cdr3_bb_mask)
