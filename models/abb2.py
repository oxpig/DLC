"""Thin adapter around the public `ImmuneBuilder` package's ABodyBuilder2
structure module. Ships as a 4-member ensemble (different embed_dim per
member: 128 for antibody_model_1, 256 for the other three), each with its
own independently-trained weights.

Notably, `ImmuneBuilder.models.StructureUpdate` defaults to `dropout=0.0`,
and `ABodyBuilder2.py` instantiates every member without overriding that
default - i.e. ABodyBuilder2 was trained/released with no dropout at all in
its structure module, architecturally, not just at inference time. The
dropout-LayerNorm correction formula is c(mu,sigma,q)=sqrt(q/(1+p*(mu/sigma)^2))
with p the train-time dropout rate; at p=0, c=1 identically, so the
correction is a mathematical no-op for this model - a clean negative
control rather than an empirical guess (contrast with Ibex, where a similar
but unconfirmed zero-dropout-finetuning hypothesis is inferred only from
muted/negative correction results, not from the architecture's own default).

Input convention: single 23-dim one-hot (21 aatype + 2 chain) per residue,
heavy sequence directly followed by light sequence, no separate pair
representation or residue-index offset at the chain break - relative
position is computed internally from plain array indices.

Checkpoint + code reference: ABodyBuilder2 (Abanades et al., Bioinformatics
2023), via the public `ImmuneBuilder` pip package; weights at
https://zenodo.org/record/7258553 (auto-downloaded on first use - see
README.md for setup).
"""
import torch

from ImmuneBuilder.models import StructureModule
from ImmuneBuilder.util import are_weights_ready, download_file, get_encoding

EMBED_DIM = {
    "antibody_model_1": 128,
    "antibody_model_2": 256,
    "antibody_model_3": 256,
    "antibody_model_4": 256,
}
MODEL_URLS = {
    name: f"https://zenodo.org/record/7258553/files/{name}?download=1" for name in EMBED_DIM
}
MODEL_NAMES = list(EMBED_DIM)


def load_model(weights_dir, model_name="antibody_model_1"):
    """Loads a single ensemble member. `weights_dir` is a local directory to
    download/cache the 4 Zenodo weight files in (not this package's own
    site-packages directory, to avoid writing into an installed package)."""
    import os

    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, model_name)
    if not are_weights_ready(weights_path):
        download_file(MODEL_URLS[model_name], weights_path)

    model = StructureModule(rel_pos_dim=64, embed_dim=EMBED_DIM[model_name])
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model.eval()


def load_ensemble(weights_dir):
    return {name: load_model(weights_dir, name) for name in MODEL_NAMES}


def correction_sites(model):
    """Both-sites convention (matching ABB3): each of the 8 layers'
    IPA-output LayerNorm (norm1) and post-residual-transition LayerNorm
    (norm2), each preceded by its own Dropout module - here always
    Dropout(p=0.0), so q=1-p=1.0 and the correction formula reduces to the
    identity (c=1) regardless of live activation statistics."""
    sites = []
    for layer in model.layers:
        sites.append((layer.norm1[0], layer.norm1[1]))
        sites.append((layer.norm2[0], layer.norm2[1]))
    return sites


def encode(sequence_dict):
    """sequence_dict: {"H": heavy_seq, "L": light_seq} -> (node_features,
    full_sequence) matching StructureModule.forward's own signature."""
    encoding = torch.tensor(get_encoding(sequence_dict), dtype=torch.get_default_dtype())
    full_seq = sequence_dict["H"] + sequence_dict["L"]
    return encoding, full_seq
