"""ABodyBuilder3's structure module: standard Invariant Point Attention
(Algorithm 22) augmented with a pair/edge representation that is
re-augmented every block with a live CA-CA distance feature computed from
the current backbone estimate. Reuses the public `openfold` package's
BackboneUpdate and rigid-body/frame utilities directly. The angle-prediction
resnet and per-block transition follow the original AlphaFold2 paper's
Algorithm 20 (a separate hidden width from the single representation's own
channel dimension, and a 2-linear residual block) rather than openfold's own
simplified reimplementation of it, so those two pieces are defined here
instead of imported.

Checkpoint + code reference: ABodyBuilder3 (Kenlay et al., arXiv:2405.20863),
weights at https://zenodo.org/records/11354577 (see README.md for setup).
"""
import math

import torch
import torch.nn as nn

from openfold.model.heads import PerResidueLDDTCaPredictor
from openfold.model.primitives import LayerNorm, Linear, ipa_point_weights_init_
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
from openfold.utils.tensor_utils import dict_multimap, flatten_final_dims, permute_final_dims


def is_fp16_enabled() -> bool:
    return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16


class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        super().__init__()
        self.linear_2 = Linear(c_hidden, c_hidden, init="relu")
        self.linear_3 = Linear(c_hidden, c_hidden, init="final")
        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        a_initial = a
        a = self.relu(a)
        a = self.linear_2(a)
        a = self.relu(a)
        a = self.linear_3(a)
        return a + a_initial


class AngleResnet(nn.Module):
    """Torsion angle prediction (Algorithm 20, lines 11-14): a residual MLP
    over both the current and initial single representations, with its own
    hidden width separate from the single representation's channel dim."""

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        super().__init__()
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(c_in, c_hidden)
        self.linear_initial = Linear(c_in, c_hidden)
        self.layers = nn.ModuleList([AngleResnetBlock(c_hidden) for _ in range(no_blocks)])
        self.linear_out = Linear(c_hidden, no_angles * 2)
        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor, s_initial: torch.Tensor):
        s_initial = self.linear_initial(self.relu(s_initial))
        s = self.linear_in(self.relu(s))
        s = s + s_initial
        for layer in self.layers:
            s = layer(s)
        s = self.linear_out(self.relu(s))

        s = s.view(s.shape[:-1] + (self.no_angles, 2))
        unnormalized_s = s
        norm_denom = torch.sqrt(torch.clamp(torch.sum(s**2, dim=-1, keepdim=True), min=self.eps))
        s = s / norm_denom
        return unnormalized_s, s


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c, c_hidden):
        super().__init__()
        self.linear_1 = Linear(c, c_hidden, init="relu")
        self.linear_2 = Linear(c_hidden, c_hidden, init="relu")
        self.linear_3 = Linear(c_hidden, c, init="final")
        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        s_initial = s
        s = self.relu(self.linear_1(s))
        s = self.relu(self.linear_2(s))
        s = self.linear_3(s)
        return s + s_initial


class StructureModuleTransition(nn.Module):
    def __init__(self, c, c_hidden, num_layers, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([StructureModuleTransitionLayer(c, c_hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm(c)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            s = layer(s)
        s = self.dropout(s)
        s = self.layer_norm(s)
        return s


class InvariantPointAttention(nn.Module):
    """Standard IPA (Algorithm 22) with a pair/edge representation term."""

    def __init__(self, c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points, inf=1e5, eps=1e-8):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        hc = c_hidden * no_heads
        self.linear_q = Linear(c_s, hc, bias=True)
        self.linear_kv = Linear(c_s, 2 * hc, bias=True)

        hpq = no_heads * no_qk_points * 3
        self.linear_q_points = Linear(c_s, hpq, bias=True)

        hpkv = no_heads * (no_qk_points + no_v_points) * 3
        self.linear_kv_points = Linear(c_s, hpkv, bias=True)

        self.linear_b = Linear(c_z, no_heads, bias=True)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = no_heads * (c_z + c_hidden + no_v_points * 4)
        self.linear_out = Linear(concat_out_dim, c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    @staticmethod
    def _point_sq_distance(q_pts: torch.Tensor, k_pts: torch.Tensor) -> torch.Tensor:
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_att**2
        return sum(torch.unbind(pt_att, dim=-1))

    def forward(self, s: torch.Tensor, z: torch.Tensor, r: Rigid, mask: torch.Tensor) -> torch.Tensor:
        q = self.linear_q(s)
        kv = self.linear_kv(s)
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        q_pts = self.linear_q_points(s)
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

        kv_pts = self.linear_kv_points(s)
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))
        k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

        b = self.linear_b(z)

        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                a = torch.matmul(
                    permute_final_dims(q.float(), (1, 0, 2)),
                    permute_final_dims(k.float(), (1, 2, 0)),
                )
        else:
            a = torch.matmul(
                permute_final_dims(q, (1, 0, 2)),
                permute_final_dims(k, (1, 2, 0)),
            )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

        pt_att = self._point_sq_distance(q_pts, k_pts)
        head_weights = self.softplus(self.head_weights).view(*((1,) * len(pt_att.shape[:-2]) + (-1, 1)))
        head_weights = head_weights * math.sqrt(1.0 / (3 * (self.no_qk_points * 9.0 / 2)))
        pt_att = pt_att * head_weights
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)
        o = flatten_final_dims(o, 2)

        o_pt = torch.sum(
            (a[..., None, :, :, None] * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]),
            dim=-2,
        )
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)
        o_pt_norm = flatten_final_dims(torch.sqrt(torch.sum(o_pt**2, dim=-1) + self.eps), 2)
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        o_pair = torch.matmul(a.transpose(-2, -3), z.to(dtype=a.dtype))
        o_pair = flatten_final_dims(o_pair, 2)

        s = self.linear_out(
            torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1).to(dtype=z.dtype)
        )
        return s


class StructureModule(nn.Module):
    """8-block structure module with a pair representation re-augmented
    every block with a fresh CA-CA distance computed from the current
    backbone estimate."""

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
        use_plddt=True,
    ):
        super().__init__()
        self.no_blocks = no_blocks
        self.trans_scale_factor = trans_scale_factor
        self.use_plddt = use_plddt

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

    def forward(self, evoformer_output_dict, aatype, mask=None, **kwargs):
        s = evoformer_output_dict["single"]
        gly_idx = restype_order_with_x["G"]
        unk_idx = restype_order_with_x["X"]
        aatype[aatype == unk_idx] = gly_idx

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


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]["model_config"]
    state_dict = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()}

    model = StructureModule(
        c_s=hp["c_s"], embed_dim=hp["embed_dim"], c_z=hp["c_z"], c_ipa=hp["c_ipa"], c_resnet=hp["c_resnet"],
        no_heads_ipa=hp["no_heads_ipa"], no_qk_points=hp["no_qk_points"], no_v_points=hp["no_v_points"],
        dropout_rate=hp["dropout_rate"], no_blocks=hp["no_blocks"], no_transition_layers=hp["no_transition_layers"],
        no_resnet_blocks=hp["no_resnet_blocks"], no_angles=hp["no_angles"], trans_scale_factor=hp["trans_scale_factor"],
        epsilon=hp["epsilon"], inf=hp["inf"], use_plddt=hp["use_plddt"],
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"
    return model.eval()


def correction_sites(model):
    """Returns the list of (dropout_module, layernorm_module) site pairs:
    the shared IPA dropout site and each block's transition site
    (both-sites convention, matching the other models in this repo)."""
    sites = [(model.ipa_dropout, model.layer_norm_ipa_layers[i]) for i in range(model.no_blocks)]
    sites += [(model.transition_layers[i].dropout, model.transition_layers[i].layer_norm) for i in range(model.no_blocks)]
    return sites
