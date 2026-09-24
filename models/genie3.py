"""Thin adapter around the public Genie3 package (aqlaboratory/genie3), a
diffusion model over backbone frames rather than a direct regression model
like the other structure predictors in this repo. Its denoiser applies the
same dropout->LayerNorm pattern per structure-net layer as the regression
models (StructureTransition.forward: `s = self.layer_norm(self.dropout(s))`,
with the transition call replacing (not added residually to) its input, so
the plain undamped correction applies directly - see
`correction.theory_c.theory_c`). See correction_sites' docstring for the
full 26-site undamped enumeration (structure_net + pair_transform_net's
own ipa/transition sub-block). The trunk's triangle-update sites are a
separate residual-bypass ("damped") pattern, not returned by
correction_sites - only the undamped sites are corrected.

Checkpoint + code reference: Genie3 (aqlaboratory/genie3),
https://github.com/aqlaboratory/genie3 - the public v1 checkpoint is
downloaded via the repo's own `scripts/setup/download.sh` into
`pretrained/v1/checkpoints/step=600000.ckpt`.
"""
import torch
from genie3.generation.model.implementation.v1 import V1Denoiser
from genie3.generation.config.model.v1 import config as model_config
from genie3.generation.diffusion.ddpm import DDPM


def load_model(checkpoint_path):
    model = V1Denoiser(model_config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    return model.eval()


def load_diffusion(n_timestep=1000, schedule="cosine", device="cpu"):
    diffusion = DDPM(n_timestep=n_timestep, schedule=schedule)
    diffusion.setup(str(device))
    return diffusion


def correction_sites(model):
    """Returns all 26 undamped (dropout->LayerNorm, no residual bypass)
    site pairs in the model: structure_net's 8 blocks x 2 sites
    (ipa_layer_norm fed by ipa_dropout, transition.layer_norm fed by
    transition.dropout) plus the earlier trunk, pair_transform_net (a
    5-block LatentTransformer - genie3.generation.model.latent.transformer.
    LatentTransformerBlock), whose blocks each contain the structurally
    identical ipa_layer_norm/transition pair (there under the name
    ipa_transition rather than transition) in addition to their own
    triangle-update sites, which are a damped residual-bypass pattern
    instead (see module docstring) - not returned here."""
    sites = []
    for layer in model.structure_net.net:
        sites.append((layer.ipa_dropout, layer.ipa_layer_norm))
        sites.append((layer.transition.dropout, layer.transition.layer_norm))
    for layer in model.pair_transform_net.net:
        sites.append((layer.ipa_dropout, layer.ipa_layer_norm))
        sites.append((layer.ipa_transition.dropout, layer.ipa_transition.layer_norm))
    return sites
