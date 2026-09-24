"""Ibex's structure module: the same standard, pair-representation IPA
architecture as ABodyBuilder3 (see models/abb3.py; both follow the original
AlphaFold2 Algorithm 20/22), extended with two things: a small MLP
(`plm_in`) that projects a per-residue ESM-C protein language model
embedding down before concatenating it onto the single representation, and
a skip connection re-adding the initial single representation after every
block but the last.

`plm_in`'s final Dropout->LayerNorm (index 4 -> index 5 below) is a second,
structurally identical dropout+LayerNorm correction site alongside the
per-block ones shared with ABB3 - see correction/theory_c.py.

Checkpoint + code reference: Ibex (Prescient Design / Genentech),
https://github.com/prescient-design/ibex (see README.md for setup).
"""
import torch
import torch.nn as nn

from openfold.model.heads import PerResidueLDDTCaPredictor
from openfold.model.primitives import LayerNorm, Linear
from openfold.model.structure_module import BackboneUpdate
from openfold.np.residue_constants import (
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_order_with_x,
    restype_rigid_group_default_frame,
)
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.utils.tensor_utils import dict_multimap

from models.abb3 import AngleResnet, InvariantPointAttention, StructureModuleTransition


class StructureModule(nn.Module):
    def __init__(
        self,
        c_s,
        embed_dim,
        c_z,
        c_ipa,
        c_resnet,
        no_heads_ipa,
        no_qk_points,
        no_v_points,
        dropout_rate,
        no_blocks,
        no_transition_layers,
        no_resnet_blocks,
        no_angles,
        trans_scale_factor,
        epsilon,
        inf,
        plm_embedding_dim,
        plm_input_dim,
        use_skip_connection=True,
        use_plddt=True,
    ):
        super().__init__()
        self.no_blocks = no_blocks
        self.trans_scale_factor = trans_scale_factor
        self.use_skip_connection = use_skip_connection
        self.use_plddt = use_plddt

        self.plm_in = nn.Sequential(
            Linear(plm_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            Linear(256, plm_input_dim),
            nn.Dropout(0.2),
            LayerNorm(plm_input_dim),
        )

        self.linear_in_node = Linear(c_s, embed_dim)
        self.linear_in_edge = Linear(c_z, embed_dim - 1)

        self.ipa_layers = nn.ModuleList(
            [
                InvariantPointAttention(embed_dim, embed_dim, c_ipa, no_heads_ipa, no_qk_points, no_v_points, inf=inf, eps=epsilon)
                for _ in range(no_blocks)
            ]
        )
        self.ipa_dropout = nn.Dropout(dropout_rate)
        self.layer_norm_ipa_layers = nn.ModuleList([LayerNorm(embed_dim) for _ in range(no_blocks)])
        self.transition_layers = nn.ModuleList(
            [StructureModuleTransition(embed_dim, c_resnet, no_transition_layers, dropout_rate) for _ in range(no_blocks)]
        )
        self.bb_update_layers = nn.ModuleList([BackboneUpdate(embed_dim) for _ in range(no_blocks)])
        self.angle_resnet_layers = nn.ModuleList(
            [AngleResnet(embed_dim, c_resnet, no_resnet_blocks, no_angles, epsilon) for _ in range(no_blocks)]
        )

        if self.use_plddt:
            self.plddt = PerResidueLDDTCaPredictor(no_bins=50, c_in=embed_dim, c_hidden=256)

    def forward(self, evoformer_output_dict, aatype, plm_embedding, mask=None, **kwargs):
        gly_idx = restype_order_with_x["G"]
        unk_idx = restype_order_with_x["X"]
        aatype[aatype == unk_idx] = gly_idx

        plm_feat = self.plm_in(plm_embedding)
        s = torch.cat((evoformer_output_dict["single"], plm_feat), dim=-1)

        if mask is None:
            mask = s.new_ones(s.shape[:-1])

        z_initial = self.linear_in_edge(evoformer_output_dict["pair"])
        s = self.linear_in_node(s)
        s_initial = s

        rigids = Rigid.identity(s.shape[:-1], s.dtype, s.device, self.training, fmt="quat")
        outputs = []
        for i in range(self.no_blocks):
            z = torch.cat(
                (z_initial, self.pairwise_distance_feature_map(rigids, mask).unsqueeze(-1).to(z_initial.dtype)),
                dim=-1,
            )
            s = s + self.ipa_layers[i](s, z, rigids, mask)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa_layers[i](s)
            s = self.transition_layers[i](s)
            if self.use_skip_connection and i < self.no_blocks - 1:
                s = s + s_initial

            rigids = rigids.compose_q_update_vec(self.bb_update_layers[i](s))
            backb_to_global = Rigid(
                Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
                rigids.get_trans(),
            )
            backb_to_global = backb_to_global.scale_translation(self.trans_scale_factor)

            unnormalized_angles, angles = self.angle_resnet_layers[i](s, s_initial)
            all_frames_to_global = self.torsion_angles_to_frames(backb_to_global, angles, aatype)
            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(all_frames_to_global, aatype)
            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

            outputs.append(
                {
                    "frames": scaled_rigids.to_tensor_7(),
                    "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                    "unnormalized_angles": unnormalized_angles,
                    "angles": angles,
                    "positions": pred_xyz,
                    "states": s,
                }
            )

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s
        if self.use_plddt:
            outputs["plddt"] = self.plddt(s)
        return outputs

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(restype_rigid_group_default_frame, dtype=float_dtype, device=device, requires_grad=False),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(restype_atom14_to_rigid_group, dtype=torch.long, device=device, requires_grad=False),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(restype_atom14_mask, dtype=float_dtype, device=device, requires_grad=False),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(restype_atom14_rigid_group_positions, dtype=float_dtype, device=device, requires_grad=False),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        self._init_residue_constants(alpha.dtype, alpha.device)
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(self, r, f):
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r, f, self.default_frames, self.group_idx, self.atom_mask, self.lit_positions
        )

    @staticmethod
    def pairwise_distance_feature_map(rigids, mask):
        trans = rigids.get_trans()
        pairwise_distances = torch.norm(trans[:, :, None] - trans[:, None, :], dim=-1)
        pairwise_mask = mask[:, :, None] * mask[:, None, :]
        return pairwise_distances * pairwise_mask


def load_model(checkpoint_path, member_idx=0):
    """Ibex ships as an 8-member deep ensemble (one state_dict per member,
    all sharing one architecture/hyperparameter config) - member_idx selects
    which one to load; a real ensemble average needs all 8 loaded and
    predictions combined by the caller."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt["model_config"]
    state_dict = ckpt["state_dicts"][member_idx]

    model = StructureModule(
        c_s=hp["c_s"], embed_dim=hp["embed_dim"], c_z=hp["c_z"], c_ipa=hp["c_ipa"], c_resnet=hp["c_resnet"],
        no_heads_ipa=hp["no_heads_ipa"], no_qk_points=hp["no_qk_points"], no_v_points=hp["no_v_points"],
        dropout_rate=hp["dropout_rate"], no_blocks=hp["no_blocks"], no_transition_layers=hp["no_transition_layers"],
        no_resnet_blocks=hp["no_resnet_blocks"], no_angles=hp["no_angles"], trans_scale_factor=hp["trans_scale_factor"],
        epsilon=hp["epsilon"], inf=hp["inf"], plm_embedding_dim=hp["plm_embedding_dim"],
        plm_input_dim=hp["plm_input_dim"], use_skip_connection=hp["use_skip_connection"], use_plddt=hp["use_plddt"],
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    return model.eval()


def correction_sites(model):
    """Returns the list of (dropout_module, layernorm_module) site pairs
    for the shared structure-module architecture (IPA + per-block
    transition), matching ABB3's both-sites convention - use with
    q=1-dropout_rate (0.9 for the public checkpoint). The plm_in projection
    (see module docstring) is a separate site with its own, different
    dropout rate - see plm_correction_site."""
    sites = [(model.ipa_dropout, model.layer_norm_ipa_layers[i]) for i in range(model.no_blocks)]
    sites += [(model.transition_layers[i].dropout, model.transition_layers[i].layer_norm) for i in range(model.no_blocks)]
    return sites


def plm_correction_site(model):
    """Returns the plm_in projection's (dropout_module, layernorm_module)
    pair as a single-element list, plus its own dropout keep probability
    (read directly off the live module, not hardcoded)."""
    dropout_module, ln_module = model.plm_in[4], model.plm_in[5]
    q = 1.0 - dropout_module.p
    return [(dropout_module, ln_module)], q
