"""Thin adapter around the public `openfold` pip package (AlphaFold2
reproduction), using the `finetuning_no_templ_ptm` model config/checkpoint
(no templates, single-sequence-friendly finetuning stage) and real MSAs.

OpenFold was trained on real evolutionary MSAs, unlike the antibody-specific
models in this repo - feeding it a fake depth-1 "MSA" (just the query
sequence) makes it fold close to randomly. `build_batch` below expects a
real .a3m alignment (e.g. fetched from the public ColabFold MMseqs2 API,
mode="env") rather than trying to run a local search pipeline.

Its structure module reuses the same IPA + transition architecture as
ESMFold (both ultimately descend from the original AlphaFold2 structure
module), and like ESMFold its per-block submodules are tied/shared across
recurrent iterations - so there is exactly one ipa_dropout/layer_norm_ipa
pair and one transition.dropout/transition.layer_norm pair to hook.
"""
import numpy as np
import torch
from openfold.model.model import AlphaFold
from openfold.config import model_config
from openfold.data import parsers, feature_pipeline
from openfold.np import residue_constants


def load_model(checkpoint_path, num_recycles=0):
    cfg = model_config('finetuning_no_templ_ptm')
    cfg.data.common.use_templates = False
    cfg.data.predict.max_templates = 0
    cfg.data.common.max_recycling_iters = num_recycles

    model = AlphaFold(cfg)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    return model.eval(), cfg


def correction_sites(model):
    """Returns the list of (dropout_module, layernorm_module) site pairs:
    the shared IPA site and the shared single-representation transition
    site (both-sites convention, matching the other models in this repo)."""
    sm = model.structure_module
    return [
        (sm.ipa_dropout, sm.layer_norm_ipa),
        (sm.transition.dropout, sm.transition.layer_norm),
    ]


def _sequence_features(sequence, description):
    num_res = len(sequence)
    return {
        "aatype": residue_constants.sequence_to_onehot(
            sequence=sequence,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True,
        ),
        "between_segment_residues": np.zeros((num_res,), dtype=np.int32),
        "domain_name": np.array([description.encode("utf-8")], dtype=object),
        "residue_index": np.array(range(num_res), dtype=np.int32),
        "seq_length": np.array([num_res] * num_res, dtype=np.int32),
        "sequence": np.array([sequence.encode("utf-8")], dtype=object),
    }


def _msa_features(a3m_path, query_sequence):
    """Parses a real .a3m alignment (ColabFold's uniref+env output) into
    OpenFold's expected MSA feature format, deduplicating exact-duplicate
    rows the way openfold.data.data_pipeline.make_msa_features does."""
    text = open(a3m_path).read()
    seqs, dels = parsers.parse_a3m(text)
    assert len(seqs[0]) == len(query_sequence), (
        f"a3m column count {len(seqs[0])} != query length {len(query_sequence)}"
    )
    seen = set()
    int_msa, deletion_matrix = [], []
    for seq, delvec in zip(seqs, dels):
        if seq in seen:
            continue
        seen.add(seq)
        int_msa.append([residue_constants.HHBLITS_AA_TO_ID[r] for r in seq])
        deletion_matrix.append(delvec)
    num_res = len(query_sequence)
    return {
        "deletion_matrix_int": np.array(deletion_matrix, dtype=np.int32),
        "msa": np.array(int_msa, dtype=np.int32),
        "num_alignments": np.array([len(int_msa)] * num_res, dtype=np.int32),
        "msa_species_identifiers": np.array([b""] * len(int_msa), dtype=object),
    }


def build_batch(sequence, a3m_path, cfg, tag="query", device="cpu"):
    """Builds a model-ready input batch from a query sequence and a real
    .a3m alignment file, using OpenFold's own feature pipeline."""
    raw_features = {**_sequence_features(sequence, tag), **_msa_features(a3m_path, sequence)}
    fp = feature_pipeline.FeaturePipeline(cfg.data)
    processed = fp.process_features(raw_features, mode="predict")
    return {k: torch.as_tensor(v).unsqueeze(0).to(device) for k, v in processed.items()}
