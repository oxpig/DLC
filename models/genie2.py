"""Thin adapter around the public Genie2 package (aqlaboratory/genie2).
Shares the same top-level import name ("genie") as Genie1 - the two
CANNOT be pip-installed editable in the same venv/site-packages at once.
Callers must instead put genie2's repo root first on PYTHONPATH (ahead of
whatever venv has genie1 installed) so `import genie` resolves to this
package - see scripts/run_genie2_dlc.py for the invocation.

Genie2 keeps the same structure-net dropout->LayerNorm pattern as
Genie1/Genie3 (StructureTransition.forward: `s=self.layer_norm(self.
dropout(s))`), with n_structure_layer=8 (like Genie3, unlike Genie1's 5 -
confirmed against the actual published epoch=40.ckpt, not just config
defaults). Its pre-structure-net trunk (PairTransformNet) is functionally
identical to Genie1's (same triangle-update-into-PairTransition residual-
bypass pattern - see models/genie1.py's module docstring; only the undamped
sites below are corrected), just reads its mask from features['residue_mask']
instead of taking it as a direct forward() argument.

Checkpoint reference: Genie2 (aqlaboratory/genie2) v1.0.0 release,
epoch=40.ckpt - the "unconditional generation" checkpoint per the repo's
own README (epoch 30 is the motif-scaffolding checkpoint instead).
"""
import torch

from genie.config import Config
from genie.diffusion.genie import Genie


def load_model(config_path, checkpoint_path, device="cpu"):
    config = Config(str(config_path))
    model = Genie.load_from_checkpoint(
        str(checkpoint_path), config=config, map_location=device
    )
    return model.eval().to(device)


def correction_sites(model):
    """Undamped sites: one pair per structure-net layer's IPA block, one
    pair per its transition block. 8 layers in the published checkpoint
    -> 16 site pairs (unlike Genie1's 10 - see models/genie1.py)."""
    sites = []
    for layer in model.model.structure_net.net:
        sites.append((layer.ipa_dropout, layer.ipa_layer_norm))
        sites.append((layer.transition.dropout, layer.transition.layer_norm))
    return sites
