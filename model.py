"""SDSC-Net: Transformer encoder-decoder with SDSA and SFCM.


"""

import torch
import torch.nn as nn

from pointnet_util import (
    PointNetFeaturePropagation,
    farthest_point_sample,
    index_points,
)
from .transformer import TransformerBlock, GlobalSemanticFeatureCalibration


def _cfg_get(node, key, default=None):
    """Read an optional field from dict / OmegaConf / EasyDict / namespace configs."""
    if node is None:
        return default
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            value = getter(key, default)
            return default if value is None else value
        except Exception:
            pass
    value = getattr(node, key, default)
    return default if value is None else value


def _build_semantic_calibration(cfg, channels):
    # Keep the supplied SFCM settings and residual calibration mechanism.
    return GlobalSemanticFeatureCalibration(
        dim=channels,
        calibration_dim=int(_cfg_get(cfg.model, "gsfc_dim", 32)),
        num_anchors=int(_cfg_get(cfg.model, "gsfc_anchors", 16)),
        tau_init=float(_cfg_get(cfg.model, "gsfc_tau_init", 0.2)),
        tau_min=float(_cfg_get(cfg.model, "gsfc_tau_min", 0.05)),
        eps=float(_cfg_get(cfg.model, "gsfc_eps", 1e-6)),
        gate_bias_init=float(_cfg_get(cfg.model, "gsfc_gate_bias", -2.0)),
    )


def resolve_spectral_slice(num_channels, override=None):
    """Locate spectral channels after XYZ; labels are not input attributes.

    Native inputs: NUIST C=8 -> 3:8, WS C=6 -> 3:6.
    Existing feature layouts remain supported: C=9 -> 3:8, C=10 -> 3:6.
    An explicit cfg.model.spectral_range=(start, end) takes precedence.
    Any additional input columns must be actual features, not target labels.
    """
    if override is not None:
        start, end = int(override[0]), int(override[1])
        if not (3 <= start < end <= num_channels):
            raise ValueError(
                f"Invalid spectral_range {(start, end)} for {num_channels} input channels"
            )
        return start, end
    if num_channels in (8, 9):
        return 3, 8
    if num_channels == 10:
        return 3, 6
    if num_channels >= 6:
        return 3, 6
    return None



class TransitionDown(nn.Module):
    """Downsample only; no PointNet++ grouping and no DSNS.

    The original encoder-decoder needs fewer points at deeper Transformer
    stages.  We retain that hierarchy with FPS and project the selected point
    features to the next channel width.  This is deliberately not a vanilla
    PointNet++ Set Abstraction block: there is no local MLP/max-pool and no
    spectral neighbor selection.
    """

    def __init__(self, k, nneighbor, channels):
        super().__init__()
        del nneighbor  # kept in the constructor for compatibility
        in_dim = channels[0] - 3
        out_dim = channels[-1]
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.npoint = int(k)

    def forward(self, xyz, points, spectral=None):
        # No SpectralSelection/DSNS code is reached from this path.  ``spectral``
        # (B x N x C_s) is only index-subsampled with the same FPS indices as
        # xyz, so that every deeper Transformer level still sees the spectral
        # values that belong to *its own* points.
        sample_count = min(self.npoint, xyz.size(1))
        if sample_count <= 0:
            raise ValueError("TransitionDown requires a positive sample count")

        fps_idx = farthest_point_sample(xyz, sample_count)
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)
        new_points = self.proj(new_points)
        new_spectral = index_points(spectral, fps_idx) if spectral is not None else None
        return new_xyz, new_points, new_spectral


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(
            xyz2.transpose(1, 2),
            xyz1.transpose(1, 2),
            None,
            feats1.transpose(1, 2),
        ).transpose(1, 2)
        return feats1 + feats2


class Backbone(nn.Module):
    """Transformer encoder; spectral values follow the same sampled points."""

    def __init__(self, cfg):
        super().__init__()
        npoints = cfg.num_point
        nblocks = cfg.model.nblocks
        nneighbor = cfg.model.nneighbor
        d_points = cfg.input_dim

        # ---- spectral difference encoding (SDE) configuration ------------------
        # Optional fields under cfg.model (all have defaults):
        #   use_spectral_diff   : bool, default True   (False -> original attention)
        #   spectral_range      : [start, end], default None (auto by channel count)
        #   spectral_scale      : float, default 1.0   (e.g. 1/255 for 0~255 data)
        #   spectral_normalize  : bool, default True   (per-band running normalisation)
        self.spectral_override = _cfg_get(cfg.model, "spectral_range", None)
        spectral_slice = resolve_spectral_slice(d_points, self.spectral_override)
        use_sde = bool(_cfg_get(cfg.model, "use_spectral_diff", True))
        self.spectral_dim = 0 if spectral_slice is None else spectral_slice[1] - spectral_slice[0]
        self.spectral_kwargs = dict(
            spectral_dim=self.spectral_dim if use_sde else 0,
            spectral_scale=float(_cfg_get(cfg.model, "spectral_scale", 1.0)),
            spectral_normalize=bool(_cfg_get(cfg.model, "spectral_normalize", True)),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.transformer1 = TransformerBlock(
            32, cfg.model.transformer_dim, nneighbor, **self.spectral_kwargs
        )

        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(
                    npoints // 4 ** (i + 1),
                    nneighbor,
                    [channel // 2 + 3, channel, channel],
                )
            )
            self.transformers.append(
                TransformerBlock(
                    channel, cfg.model.transformer_dim, nneighbor, **self.spectral_kwargs
                )
            )

        self.nblocks = nblocks

    def forward(self, x):
        if x.ndim != 3 or x.size(-1) < 3:
            raise ValueError("Expected point cloud with shape B x N x C, C >= 3")
        if x.size(-1) != self.fc1[0].in_features:
            raise ValueError(
                "Input has {} channels, but cfg.input_dim is {}. "
                "Pass feature attributes only; keep target labels separate.".format(
                    x.size(-1), self.fc1[0].in_features
                )
            )

        xyz = x[..., :3]
        spectral_slice = resolve_spectral_slice(x.size(-1), self.spectral_override)
        if spectral_slice is None:
            spectral = None
        else:
            start, end = spectral_slice
            spectral = x[..., start:end]

        # Input MLP -> SDSA; no geometry-branch concatenation or fc5 projection.
        points = self.fc1(x)
        points = self.transformer1(xyz, points, spectral)[0]
        xyz_and_feats = [(xyz, points, spectral)]
        for transition_down, transformer in zip(self.transition_downs, self.transformers):
            xyz, points, spectral = transition_down(xyz, points, spectral)
            points = transformer(xyz, points, spectral)[0]
            xyz_and_feats.append((xyz, points, spectral))
        return points, xyz_and_feats


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        nblocks = cfg.model.nblocks
        n_c = cfg.num_class
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c),
        )

    def forward(self, x):
        points, _ = self.backbone(x)
        return self.fc2(points.mean(1))


class PointTransformerSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        nblocks = cfg.model.nblocks
        nneighbor = cfg.model.nneighbor
        n_c = cfg.num_class
        final_dim = 32 * 2 ** nblocks

        self.fc2 = nn.Sequential(
            nn.Linear(final_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, final_dim),
        )
        self.transformer2 = TransformerBlock(
            final_dim,
            cfg.model.transformer_dim,
            nneighbor,
            **self.backbone.spectral_kwargs,
        )
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(
                TransformerBlock(
                    channel,
                    cfg.model.transformer_dim,
                    nneighbor,
                    **self.backbone.spectral_kwargs,
                )
            )

        # The final decoder/SFCM feature is 32-D at the original resolution.
        self.fc3 = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_c),
        )
        # The segmentation branch has C=32 after its final upsampling/attention.
        self.use_gsfc = bool(_cfg_get(cfg.model, "use_gsfc", True))
        self.semantic_calibration = _build_semantic_calibration(cfg, 32)

    def forward(self, x, return_feat=False, return_branches=False, return_aux=False):
        """Return B x N x num_class logits for cross-entropy or argmax.

        return_feat: also return the final B x N x 32 calibrated features.
        return_branches: diagnostic dict containing only "spectral_spatial".
        return_aux: (logits, {}); no geometry auxiliary task remains.
        """
        if return_aux and (return_feat or return_branches):
            raise ValueError("Use return_aux separately from feature/branch returns")
        points, xyz_and_feats = self.backbone(x)
        xyz, _, spectral = xyz_and_feats[-1]
        points = self.transformer2(xyz, self.fc2(points), spectral)[0]

        for i in range(self.nblocks):
            skip_xyz, skip_points, skip_spectral = xyz_and_feats[-i - 2]
            points = self.transition_ups[i](xyz, points, skip_xyz, skip_points)
            xyz = skip_xyz
            points = self.transformers[i](xyz, points, skip_spectral)[0]

        # SFCM is applied once after the last decoder SDSA block, as in the paper.
        # Its internal residual gate is part of SFCM, not dual-branch fusion.
        if self.use_gsfc:
            points = self.semantic_calibration(points)
        logits = self.fc3(points)

        if return_aux:
            return logits, {}
        if return_branches:
            branches = {"spectral_spatial": points}
            if return_feat:
                return logits, points, branches
            return logits, branches
        if return_feat:
            return logits, points
        return logits
