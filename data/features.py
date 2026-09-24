"""Relative-position pair-representation features shared by ABB3, Ibex, and
NbForge - pure torch, no model-package dependency, so it's usable from any
model's environment regardless of whether that environment has `openfold`
installed.
"""
import torch

REL_POS_DIM = 64
PAIR_DIM = 2 * REL_POS_DIM + 1  # relative-position one-hot, single chain
PAIR_DIM_PAIRED = 3 + PAIR_DIM  # +3 for the chain (heavy/light) one-hot, paired VH/VL


def pair_features(residue_index: torch.Tensor) -> torch.Tensor:
    """(n,) residue indices -> (n, n, PAIR_DIM) one-hot relative position."""
    pair = residue_index[None, :] - residue_index[:, None]
    pair = pair.clamp(-REL_POS_DIM, REL_POS_DIM) + REL_POS_DIM
    return torch.nn.functional.one_hot(pair, PAIR_DIM).float()


def pair_features_paired(residue_index: torch.Tensor, is_heavy: torch.Tensor) -> torch.Tensor:
    """(n,) residue indices + (n,) is_heavy (1=heavy chain, 0=light chain, or
    all-ones for a single-chain/nanobody input) -> (n, n, PAIR_DIM_PAIRED):
    chain one-hot concatenated with the relative-position one-hot (chain
    feature first, then relative position)."""
    rel_pos = pair_features(residue_index)
    chain_class = 2 * is_heavy.outer(is_heavy) + (1 - is_heavy).outer(1 - is_heavy)  # (n, n) in {0,1,2}
    chain_onehot = torch.nn.functional.one_hot(chain_class.long(), 3).float()
    return torch.cat((chain_onehot, rel_pos), dim=-1)
