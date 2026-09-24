"""Thin adapter around the public QuickBind package (aqlaboratory/QuickBind,
https://github.com/aqlaboratory/QuickBind), a light-weight molecular
docking model sharing the same weight-tied IPA structure-module pattern as
ESMFold/OpenFold/Genie3 (a single structure-module block applied
repeatedly - here 8 times - rather than one block per layer), regressing
ligand-atom coordinates directly (no diffusion). Only 2 correction sites:
the shared block's `ipa_dropout`->`layer_norm_ipa` and
`transition.dropout`->`transition.layer_norm`.

Checkpoint + code reference: QuickBind (Treyde, Kim, Bouatta, AlQuraishi,
arXiv:2410.16474), weights bundled in the repo at
checkpoints/quickbind_default/{config.yaml,best_checkpoint.pt}.

QuickBind's own dependency stack is unusually old and fragile - pinned to
torch==1.12.1+cu113, plus a pinned `aqlaboratory/openfold` fork (commit
efcf80f50e5534cdf9b5ede56ef1cbbd1471775e, NOT the public `openfold` pip
package used by every other model in this repo) for its structure-module
primitives. On a modern system, expect to need: (1) a CPU/pure-PyTorch stub
for the uncompiled `attn_core_inplace_cuda` CUDA extension (or a working
CUDA 11.3 + gcc<=10 toolchain to actually compile it), (2) no-op stubs for
`deepspeed` and `dllogger` (only used for training-time logging/checkpoint
gating, never on the inference path used here), and (3) possibly an
executable-stack fix for the old torch build's `libtorch_cpu.so` on
kernels with modern stack-hardening (a `PT_GNU_STACK` ELF header patch).
See requirements/quickbind.txt.
"""
import torch


def load_model(quickbind_dir):
    """quickbind_dir: a local clone of aqlaboratory/QuickBind (repo root),
    already on sys.path, with checkpoints/quickbind_default/ present."""
    import yaml
    from quickbind import QuickBind
    from dataset.dataimporter import DataImporter

    with open(f"{quickbind_dir}/checkpoints/quickbind_default/config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["dataset_params"]["cropping"] = False

    # feature dims depend on the dataset's own feature construction, so a
    # DataImporter instance is needed just to read them off, even though
    # we don't use its data here
    test_data = DataImporter(
        complex_names_path=f"{quickbind_dir}/data/posebusters_benchmark_set/posebusters",
        **cfg["dataset_params"],
    )
    lig_feat_dim, rec_feat_dim = test_data.get_feature_dimensions()

    model = QuickBind(
        aa_feat=rec_feat_dim, lig_atom_feat=lig_feat_dim, **cfg["model_parameters"],
        chunk_size=2, output_s=False,
    )
    ckpt = torch.load(f"{quickbind_dir}/checkpoints/quickbind_default/best_checkpoint.pt", map_location="cpu")
    state_dict = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    return model.eval(), cfg


def correction_sites(model):
    """Returns the (dropout_module, layernorm_module) site pair for the
    single, weight-tied structure-module block (applied repeatedly, same
    pattern as ESMFold/OpenFold/Genie3 - so this is a 2-element list, not
    one pair per iteration)."""
    sm = model.structure_module_block
    return [(sm.ipa_dropout, sm.layer_norm_ipa), (sm.transition.dropout, sm.transition.layer_norm)]
