"""Point Transformer block with spectral-difference-encoding (SDE) enhanced attention.

Implements section "B. 光谱差异编码增强的点自注意力模块":

    q_i = W_q x_i,  k_j = W_k x_j,  v_j = W_v x_j
    e^p_ij = MLP_p(p_i - p_j)                     (relative position encoding)
    d^s_ij = | s_i - s_j |                        (band-wise absolute spectral difference)
    e^s_ij = MLP_s(d^s_ij)                        (spectral difference encoding)
    u_ij   = MLP_a(q_i - k_j + e^p_ij + e^s_ij)   (neighbourhood relation score)
    a_ij   = softmax_{m in N(i)}(u_im)            (normalised over neighbours)
    x'_i   = W_o( sum_j a_ij ⊙ (v_j + e^p_ij) ) + x_i

If ``spectral_dim == 0`` the block is exactly the original Point Transformer block.
"""

from pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from contextlib import nullcontext


class SpectralDifferenceEncoder(nn.Module):
    """e^s_ij = MLP_s(|s_i - s_j|).

    ``|s_i - s_j|`` is computed independently for every spectral band, so the
    input of MLP_s has ``spectral_dim`` channels (3 for RGB, 5 for
    RGB + RedEdge + IR).

    Scale handling: neighbouring points usually differ only slightly in
    reflectance, and different bands (RGB vs. RedEdge/IR) live on different
    numeric ranges.  Without any normalisation the encoder input is tiny (or
    huge for 0~255 data) and the branch contributes almost nothing to the
    attention score.  ``normalize=True`` divides each band's difference by a
    running estimate (EMA) of that band's mean neighbourhood difference, so the
    encoder always sees O(1) inputs regardless of data units.  A fixed
    ``input_scale`` (e.g. 1/255) is applied first.
    """

    def __init__(self, spectral_dim, d_model, input_scale=1.0, normalize=True,
                 momentum=0.01, eps=1e-6):
        super().__init__()
        self.spectral_dim = int(spectral_dim)
        self.input_scale = float(input_scale)
        self.normalize = bool(normalize)
        self.momentum = float(momentum)
        self.eps = float(eps)

        self.mlp = nn.Sequential(
            nn.Linear(self.spectral_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.register_buffer("running_scale", torch.ones(self.spectral_dim))
        self.register_buffer("scale_initialized", torch.zeros(1, dtype=torch.long))

    def forward(self, spectral_center, spectral_neighbor):
        # spectral_center: B x N x 1 x C_s, spectral_neighbor: B x N x k x C_s
        diff = torch.abs(spectral_center - spectral_neighbor) * self.input_scale
        if self.normalize:
            if self.training:
                with torch.no_grad():
                    batch_scale = diff.detach().float().mean(dim=(0, 1, 2)).clamp_min(self.eps)
                    if int(self.scale_initialized.item()) == 0:
                        self.running_scale.copy_(batch_scale)
                        self.scale_initialized.fill_(1)
                    else:
                        self.running_scale.mul_(1 - self.momentum).add_(
                            batch_scale, alpha=self.momentum
                        )
            diff = diff / (self.running_scale.to(diff.dtype) + self.eps)
        return self.mlp(diff)  # B x N x k x d_model


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k, spectral_dim=0,
                 spectral_scale=1.0, spectral_normalize=True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

        self.spectral_dim = int(spectral_dim)
        if self.spectral_dim > 0:
            self.spectral_encoder = SpectralDifferenceEncoder(
                self.spectral_dim, d_model,
                input_scale=spectral_scale, normalize=spectral_normalize,
            )
        else:
            self.spectral_encoder = None

    # xyz: b x n x 3, features: b x n x f, original_spectral: b x n x C_s
    # NOTE: original_spectral must be the spectral values of exactly the points
    # in ``xyz`` (i.e. subsampled together with xyz after FPS).
    def forward(self, xyz, features, original_spectral=None):
        n = xyz.size(1)
        dists = square_distance(xyz, xyz)  # b x n x n
        knn_idx = dists.topk(min(self.k, n), dim=-1, largest=False)[1]  # b x n x k (ascending)
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        # relation = (q_i - k_j) + e^p_ij  [+ e^s_ij]
        relation = q[:, :, None] - k + pos_enc

        if self.spectral_encoder is not None:
            if original_spectral is None:
                raise ValueError(
                    "TransformerBlock was built with spectral_dim="
                    f"{self.spectral_dim} but no spectral tensor was given."
                )
            if original_spectral.shape[:2] != xyz.shape[:2]:
                raise ValueError(
                    "Spectral tensor and xyz must share the same points: "
                    f"{tuple(original_spectral.shape[:2])} vs {tuple(xyz.shape[:2])}. "
                    "Subsample the spectral values together with xyz."
                )
            if original_spectral.size(-1) != self.spectral_dim:
                raise ValueError(
                    f"Expected {self.spectral_dim} spectral bands, "
                    f"got {original_spectral.size(-1)}"
                )
            spec_nb = index_points(original_spectral, knn_idx)  # b x n x k x C_s
            spec_enc = self.spectral_encoder(original_spectral[:, :, None], spec_nb)
            relation = relation + spec_enc

        attn = self.fc_gamma(relation)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # normalised over the k neighbours

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn


# B:  semantic feature calibration (B1--B5).
def _autocast_disabled(device_type):
    # CUDA branch supports the project's PyTorch 1.9 environment.
    if device_type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    if device_type == "cpu" and hasattr(torch, "autocast"):
        return torch.autocast(device_type="cpu", enabled=False)
    return nullcontext()


class GlobalSemanticFeatureCalibration(nn.Module):
    """Cosine routing -> dynamic prototypes -> gated residual correction.

    Only routing uses unit-length vectors. Prototype values use the original
    projected features, as required by B3. Neither point labels nor coordinates
    enter this module. ``out_proj`` has no bias and starts at zero, so inserting
    it into a model initially preserves that model's features exactly.
    """

    def __init__(self, dim, calibration_dim=32, num_anchors=16,
                 tau_init=0.2, tau_min=0.05, eps=1e-6, gate_bias_init=0.0):
        super().__init__()
        self.dim = int(dim)
        self.calibration_dim = int(calibration_dim)
        self.num_anchors = int(num_anchors)
        self.tau_min = float(tau_min)
        self.eps = float(eps)
        tau_init = float(tau_init)
        if min(self.dim, self.calibration_dim, self.num_anchors) < 1:
            raise ValueError("dim, calibration_dim and num_anchors must be positive")
        if not all(math.isfinite(v) for v in (tau_init, self.tau_min, self.eps)):
            raise ValueError("Temperature and epsilon must be finite")
        if not (tau_init > self.tau_min > 0 and self.eps > 0):
            raise ValueError("Require tau_init > tau_min > 0 and eps > 0")

        d = self.calibration_dim
        self.norm = nn.LayerNorm(self.dim)
        self.value_proj = nn.Linear(self.dim, d, bias=False)
        self.anchors = nn.Parameter(torch.empty(self.num_anchors, d))
        nn.init.normal_(self.anchors, mean=0.0, std=d ** -0.5)

        # Stable inverse softplus: softplus(theta) = tau_init - tau_min.
        gap = tau_init - self.tau_min
        theta = gap + math.log(-math.expm1(-gap))
        self.raw_temperature = nn.Parameter(torch.tensor(theta))
        self.gate = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[2].bias, float(gate_bias_init))  # 新增：门控初始偏置
        self.out_proj = nn.Linear(d, self.dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    @property
    def temperature(self):
        return self.tau_min + F.softplus(self.raw_temperature)

    def forward(self, features, return_details=False):
        if features.ndim != 3 or features.size(-1) != self.dim:
            raise ValueError("Expected B x N x {} features".format(self.dim))
        if features.size(0) == 0 or features.size(1) == 0:
            raise ValueError("GSFC requires nonempty clouds")

        # B1: a separate, bias-free projection into the calibration space.
        values = self.value_proj(self.norm(features))
        # A cast alone is insufficient inside AMP: disable autocast around
        # matrix products as well. Preserve float64 for numerical verification.
        work_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
        with _autocast_disabled(values.device.type):
            v = values.to(work_dtype)
            a = self.anchors.to(work_dtype)
            tau = self.tau_min + F.softplus(self.raw_temperature.to(work_dtype))
            v_unit = F.normalize(v, p=2, dim=-1, eps=self.eps)
            a_unit = F.normalize(a, p=2, dim=-1, eps=self.eps)
            # B2: normalize over anchors, independently for every point.
            assignment = F.softmax(torch.matmul(v_unit, a_unit.t()) / tau, dim=-1)
            mass = assignment.sum(dim=1)  # B x M; never sum over the batch.
            # B3: aggregate unnormalized v, then broadcast prototypes to points.
            prototypes = torch.bmm(assignment.transpose(1, 2), v)
            prototypes = prototypes / mass.clamp_min(self.eps).unsqueeze(-1)
            reference = torch.bmm(assignment, prototypes)
            difference = reference - v

        # B4: all three gate inputs have dimension d (not mixed C/d inputs).
        reference_local = reference.to(values.dtype)
        difference_local = difference.to(values.dtype)
        gate = self.gate(torch.cat(
            (values, reference_local, difference_local.abs()), dim=-1
        ))
        # B5: calibrate only the difference; preserve the original residual.
        correction = self.out_proj(gate * difference_local)
        calibrated = features + correction.to(features.dtype)
        if not return_details:
            return calibrated

        # Opt-in diagnostics: detached, not retained on the module or in a
        # checkpoint. Do not request these large tensors for every train step.
        mean_assignment = mass / features.size(1)
        effective_anchors = torch.exp(-(
            mean_assignment * mean_assignment.clamp_min(self.eps).log()
        ).sum(dim=-1))
        point_entropy = -(
            assignment * assignment.clamp_min(self.eps).log()
        ).sum(dim=-1).mean(dim=-1)
        if self.num_anchors > 1:
            point_entropy = point_entropy / math.log(self.num_anchors)
        details = {
            "assignment": assignment.detach(),
            "anchor_mass": mass.detach(),
            "prototypes": prototypes.detach(),
            "reference": reference.detach(),
            "difference": difference.detach(),
            "gate": gate.detach(),
            "temperature": tau.detach(),
            "effective_anchors": effective_anchors.detach(),
            # 1 means uniform routing when M>1. Read this together with usage:
            # high effective_anchors alone does not imply informative routing.
            "normalized_point_entropy": point_entropy.detach(),
        }
        return calibrated, details
