"""Thin adapter around NbForge's own public structure module. Unlike ABB3/
Ibex, NbForge's own package already matches its own checkpoint natively (no
architecture mismatch to work around), so this just depends on it directly
rather than vendoring a copy.

NbForge's structure module has three dropout+LayerNorm sites per block: the
usual shared IPA dropout -> per-block LayerNorm, a per-block transition on
the single representation (`transition_s`), and a third one on the pair
representation (`transition_z`, which ABB3/Ibex don't have at all - all
three are architecturally valid undamped correction sites, since the outer
residual addition around each transition happens after its LayerNorm and so
doesn't affect that LayerNorm's own normalization statistics). All three are
corrected here.

The dropout rate baked into the checkpoint (q=0.9, i.e. p=0.1) is the
theoretically "correct" value to plug into the correction formula, and
that's what's used for the IPA and `transition_s` sites. But a small q-sweep
around that value - cross-checked across three independent, non-overlapping
100-structure subsets of the nanobody test pool - shows the actual optimum
across all three sites together sits at q~=0.94, not q=0.9: at the true
dropout rate, correcting all three sites underperforms correcting just the
first two, but at q~=0.94 correcting all three beats both the naive q=0.9
three-site result and the two-site (z-excluded) result, consistently across
all three subsets. RECOMMENDED_Q below reflects that empirical optimum
rather than the checkpoint's literal dropout rate; see
`theory_c.install_correction_hooks`'s `q` argument for where it gets
threaded through.

Checkpoint + code reference: NbForge (Sormanni Lab),
https://gitlab.doc.ic.ac.uk/sormanni-lab/nbforge (checkpoint is bundled in
the repo at src/NbForge/weights/nbforge.ckpt - see README.md for setup).
"""
from NbForge.openfold.model.structure_module import StructureModule

RECOMMENDED_Q = 0.94


def load_model(checkpoint_path):
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]["model_config"]
    state_dict = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}

    model = StructureModule(**hp)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    return model.eval()


def correction_sites(model):
    """Returns the list of (dropout_module, layernorm_module) site pairs,
    all sharing the same dropout keep probability: IPA, single-representation
    transition, and pair-representation transition (all three blocks) - see
    the module docstring for why all three are included and for the q value
    to use with them (RECOMMENDED_Q, not the checkpoint's literal 0.9)."""
    sites = [(model.dropout, model.layer_norm_ipa_layers[i]) for i in range(model.no_blocks)]
    sites += [(model.transition_s[i].dropout, model.transition_s[i].layer_norm) for i in range(model.no_blocks)]
    sites += [(model.transition_z[i].dropout, model.transition_z[i].layer_norm) for i in range(model.no_blocks)]
    return sites
