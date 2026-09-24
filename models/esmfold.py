"""Thin adapter around the public `esm` package's ESMFold v1 model
(esm.pretrained.esmfold_v1(), the ESM2-3B-backbone folding model). ESMFold's
structure module reuses the same IPA + per-block transition architecture as
AlphaFold2, but unlike ABB3/Ibex/NbForge its structure-module submodules
have tied weights across the 8 recurrent iterations - so there is exactly
one `ipa_dropout`/`layer_norm_ipa` pair and one `transition.dropout`/
`transition.layer_norm` pair to hook, not one per block.

Structure prediction itself goes through the model's own high-level
`model.infer(sequence, num_recycles=...)` API - callers pass a single
sequence for a monomer/nanobody, or two sequences joined with ":" for a
paired heavy:light Fv (ESMFold's own multimer convention: a glycine linker
is inserted internally and residue numbering is offset across the chain
break, so no special handling is needed at the call site).
"""
import esm


def load_model():
    return esm.pretrained.esmfold_v1().eval()


def correction_sites(model):
    """Returns the list of (dropout_module, layernorm_module) site pairs:
    the shared IPA site and the shared single-representation transition
    site (both-sites convention, matching the other models in this repo)."""
    sm = model.trunk.structure_module
    return [
        (sm.ipa_dropout, sm.layer_norm_ipa),
        (sm.transition.dropout, sm.transition.layer_norm),
    ]
