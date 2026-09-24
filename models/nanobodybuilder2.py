"""Thin adapter around the public `ImmuneBuilder` package's NanoBodyBuilder2
structure module - architecturally identical to ABodyBuilder2's (see
models/abb2.py), just trained on nanobody-only (single heavy-chain) data and
shipped as a separate 4-member ensemble/checkpoint set. Same `dropout=0.0`
default (never overridden), so the correction is an exact no-op here too.

Input convention: single 23-dim one-hot (21 aatype + 2 chain, chain always
0 since there's only one chain) per residue, heavy sequence only (no light
chain).

Checkpoint + code reference: NanoBodyBuilder2 (Abanades et al., Bioinformatics
2023), via the public `ImmuneBuilder` pip package; weights at
https://zenodo.org/record/7258553 (auto-downloaded on first use).
"""
import torch

from ImmuneBuilder.models import StructureModule
from ImmuneBuilder.util import are_weights_ready, download_file, get_encoding

EMBED_DIM = {
    "nanobody_model_1": 128,
    "nanobody_model_2": 256,
    "nanobody_model_3": 256,
    "nanobody_model_4": 256,
}
MODEL_URLS = {
    name: f"https://zenodo.org/record/7258553/files/{name}?download=1" for name in EMBED_DIM
}
MODEL_NAMES = list(EMBED_DIM)


def load_model(weights_dir, model_name="nanobody_model_1"):
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
    sites = []
    for layer in model.layers:
        sites.append((layer.norm1[0], layer.norm1[1]))
        sites.append((layer.norm2[0], layer.norm2[1]))
    return sites


def encode(heavy_sequence):
    encoding = torch.tensor(get_encoding({"H": heavy_sequence}, "H"), dtype=torch.get_default_dtype())
    return encoding, heavy_sequence
