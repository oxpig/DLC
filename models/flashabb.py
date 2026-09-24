"""Thin adapter around the public `flash_abb` package (oxpig/FlashABB), an
accelerated variant of ABodyBuilder3's structure module. Its checkpoint is
bundled in the package itself, and its correction sites follow the exact
same both-sites convention as ABB3 (models/abb3.py): a shared
`ipa_dropout` paired with each block's own `layer_norm_ipa_layers[i]`, and
each block's own `transition_layers[i].dropout`/`.layer_norm`.

Unlike ABB3, the structure module's forward pass takes no separate pair
representation - relative-position information is computed internally by
each IPA layer directly from `res_idx` - and `res_idx` itself is a required
positional argument: `model({"single": single}, aatype, res_idx, mask=...)`.
`single` is the same aatype+chain one-hot convention as ABB3
(`data.features` isn't needed here). Paired heavy+light input uses a +500
residue-index offset at the chain boundary (see the public `featurize()`
helper in `flash_abb.model.flash_abb` for the reference convention);
single-chain (nanobody) input needs no offset. Best suited to paired input -
this is fundamentally the same architecture as ABB3, a paired-antibody
model, and performs comparatively poorly on nanobody-only structures.

Checkpoint + code reference: FlashABB, https://github.com/oxpig/FlashABB
(pip install flash-abb).
"""


def load_model(device="cpu"):
    from flash_abb.load_model import load_model as _load_model

    flabb, _ = _load_model("flash-abb", device=device)
    model = flabb.model.to(device)
    model.bb_bias_scale = 1.0
    model.bb_output_scale = 1.0
    return model.eval()


def correction_sites(model):
    """Returns the list of (dropout_module, layernorm_module) site pairs:
    the shared IPA dropout site and each block's transition site
    (both-sites convention, matching ABB3's own)."""
    sites = [(model.ipa_dropout, model.layer_norm_ipa_layers[i]) for i in range(model.no_blocks)]
    sites += [(model.transition_layers[i].dropout, model.transition_layers[i].layer_norm) for i in range(model.no_blocks)]
    return sites
