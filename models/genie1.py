"""Thin adapter around the public Genie1 package (aqlaboratory/genie), the
predecessor to Genie3 (see genie3.py). Shares the same structure-module
dropout->LayerNorm pattern per structure-net layer (StructureTransition.
forward: `s = self.layer_norm(self.dropout(s))`). Its pre-structure-net
trunk (PairTransformNet) differs from Genie3's LatentTransformer - it only
refines the pair representation (no single-representation IPA/transition
sub-block) - and has its own triangle-update dropout sites feeding a
shared undropped running pair representation before PairTransition's own
LayerNorm, a residual-bypass ("damped") pattern rather than a direct
dropout->LN site. Only the undamped sites below are corrected.

Checkpoint + code reference: Genie1 (aqlaboratory/genie),
https://github.com/aqlaboratory/genie - published weights are checked
into the repo itself under weights/<name>/{configuration,epoch=*.ckpt}.
Confirmed empirically (not just from config defaults) that all three
published checkpoints (scope_l_128, scope_l_256, swissprot_l_256) use
n_structure_layer=5 (not 8, unlike Genie2/Genie3) and
n_pair_transform_layer=5, include_tri_att=False, tri_dropout=0.25,
ipa_dropout=0.1, structure_transition_dropout=0.1.
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
    """Undamped (dropout->LayerNorm, no bypass) sites: one pair per
    structure-net layer's IPA block, one pair per its transition block.
    5 layers in the published checkpoints -> 10 site pairs."""
    sites = []
    for layer in model.model.structure_net.net:
        sites.append((layer.ipa_dropout, layer.ipa_layer_norm))
        sites.append((layer.transition.dropout, layer.transition.layer_norm))
    return sites
