"""
DA3 (Depth Anything 3) dense feature backbone adapter for FoundationStereo.

Goal: provide the same interface as FoundationStereo's current DINO/DepthAnything feature backbone:
  input : Tensor[2, 3, H, W]  (concatenated left/right, i.e. torch.cat([image1, image2], dim=0))
  output: {"out": Tensor[2, 128, H, W]}  (a dense feature map per image)

Internally DA3's backbone outputs patch tokens [Bpair, 2, N, D]. We convert tokens -> dense feature map
using DA3's DPT neck up to (and including) `output_conv1`, which yields 128 channels.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _ensure_depth_anything_3_on_syspath() -> None:
    """
    Prefer an installed `depth_anything_3`. If it's not importable, add the local repo path:
      <workspace_ws>/src/Depth-Anything-3/src
    """
    try:
        importlib.import_module("depth_anything_3")
        return
    except Exception:
        pass

    file_path = Path(__file__).resolve()
    parts = file_path.parts
    if "workspace_ws" not in parts:
        raise FileNotFoundError(f"`workspace_ws` not found in path: {file_path}")

    idx = parts.index("workspace_ws")
    workspace_root = Path(*parts[: idx + 1])  # .../workspace_ws
    da3_src = workspace_root / "src" / "Depth-Anything-3" / "src"
    if not da3_src.exists():
        raise FileNotFoundError(f"Depth-Anything-3 src path not found: {da3_src}")

    da3_src_str = str(da3_src)
    if da3_src_str not in sys.path:
        sys.path.insert(0, da3_src_str)


class DA3DenseFeatureBackbone(nn.Module):
    """
    Wrap DA3 so it can be used as a dense feature extractor in FoundationStereo.

    Notes:
    - Expects input shaped as [2, 3, H, W] (left/right concatenated along batch dim).
    - H and W must be divisible by DA3 patch size (14) unless `allow_pad_to_divisible=True`.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        device: str = "cuda",
        ref_view_strategy: str = "saddle_balanced",
        use_autocast: bool = True,
        allow_pad_to_divisible: bool = False,
        input_layout: str = "stacked_lr",
    ) -> None:
        super().__init__()
        _ensure_depth_anything_3_on_syspath()

        # Import after sys.path is prepared
        from depth_anything_3.api import DepthAnything3
        from depth_anything_3.model.da3 import DepthAnything3Net

        # NOTE:
        # - We load the model onto `device` initially.
        # - During forward, we DO NOT blindly move inputs to `device_str`.
        #   Instead, we run on the module's current device (so `.to(...)` on the parent model works).
        self.device_str = device
        self.ref_view_strategy = ref_view_strategy
        self.use_autocast = use_autocast
        self.allow_pad_to_divisible = allow_pad_to_divisible
        self.input_layout = input_layout

        # Load pretrained DA3 from local directory (HF-compatible layout).
        api = DepthAnything3.from_pretrained(model_dir).to(device)
        api.eval()

        # `api.model` is a `DepthAnything3Net` (or nested variant). We only support the basic net here.
        if not isinstance(api.model, DepthAnything3Net) and not hasattr(api.model, "backbone"):
            raise TypeError(f"Unsupported DA3 model type: {type(api.model)}")

        # Backbone returns (feats, aux_feats). Head is DPT/DualDPT.
        self._da3_api = api
        self.backbone = api.model.backbone
        self.head = api.model.head

        # For safety / clarity
        self.patch_size = int(getattr(self.head, "patch_size", 14))
        self.down_ratio = int(getattr(self.head, "down_ratio", 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: torch.Tensor with shape [2*B, 3, H, W] on any device.
               This is the FoundationStereo convention: `torch.cat([left, right], dim=0)`.
               - If `input_layout="stacked_lr"` (default): x = [left_batch; right_batch]
               - If `input_layout="paired"`: x = [L0,R0,L1,R1,...] (interleaved)
        Returns:
            dict with key "out": [2*B, 128, H, W] dense features.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x.ndim==4 ([2,3,H,W]), got shape {tuple(x.shape)}")
        if x.shape[0] % 2 != 0:
            raise ValueError(f"Expected an even batch (left/right pairs). Got B={x.shape[0]}")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3-channel RGB input. Got C={x.shape[1]}")

        # Run on the module's current device (important when the parent model is moved via `.to()`).
        input_device = x.device
        try:
            model_device = next(self.parameters()).device
        except StopIteration:
            model_device = input_device
        if x.device != model_device:
            x = x.to(model_device, non_blocking=True)

        Bimg, _, H, W = x.shape
        if (H % self.patch_size) != 0 or (W % self.patch_size) != 0:
            if not self.allow_pad_to_divisible:
                raise ValueError(
                    f"Input H,W must be divisible by patch_size={self.patch_size}. "
                    f"Got H={H}, W={W}. Consider resizing like FoundationStereo does (divider=lcm(14,16)=112)."
                )
            pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
            pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            H, W = x.shape[-2], x.shape[-1]

        # [2*B,3,H,W] -> [B, 2, 3, H, W] (stereo pair)
        Bpair = Bimg // 2
        if self.input_layout == "paired":
            # Interleaved pairs: [L0,R0,L1,R1,...]
            x_pair = x.view(Bpair, 2, 3, H, W)
        elif self.input_layout == "stacked_lr":
            # Stacked batches: [L0..L(B-1), R0..R(B-1)]
            x_left = x[:Bpair]
            x_right = x[Bpair:]
            x_pair = torch.stack([x_left, x_right], dim=1)
        else:
            raise ValueError(
                f"Unknown input_layout={self.input_layout!r}. Use 'stacked_lr' or 'paired'."
            )

        if x_pair.device.type == "cpu":
            # CPU autocast supports bfloat16; float16 is typically unsupported / not useful.
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(
            device_type=x_pair.device.type,
            dtype=autocast_dtype,
            # Keep autocast only for CUDA by default; CPU autocast can be surprising / unsupported
            # depending on operator coverage.
            enabled=bool(self.use_autocast and x_pair.device.type == "cuda"),
        ):
            feats, _aux_feats = self.backbone(
                x_pair,
                cam_token=None,
                export_feat_layers=[],
                ref_view_strategy=self.ref_view_strategy,
            )

        dense = self._tokens_to_dense_features(feats, H=H, W=W)  # [Bpair, 2, 128, H, W]
        # Back to FoundationStereo layout: [2*B, 128, H, W]
        if self.input_layout == "paired":
            dense = dense.reshape(Bimg, dense.shape[2], dense.shape[3], dense.shape[4])
        else:
            dense_left = dense[:, 0]
            dense_right = dense[:, 1]
            dense = torch.cat([dense_left, dense_right], dim=0)
        if dense.device != input_device:
            dense = dense.to(input_device, non_blocking=True)
        return {"out": dense}

    def _tokens_to_dense_features(self, feats, *, H: int, W: int) -> torch.Tensor:
        """
        Convert DA3 backbone tokens to a dense feature map using DA3 head's DPT neck.

        Args:
            feats: list/tuple of 4 entries from DA3 backbone, each like (tokens, camera_token)
                  where tokens is [Bpair, 2, N, C]
            H, W: image spatial size (must match patch grid, i.e. N == (H/ps)*(W/ps))
        Returns:
            dense features: [Bpair, 2, 128, H/down_ratio, W/down_ratio] (down_ratio usually 1)
        """
        # Import here to avoid importing DA3 unless actually used.
        from depth_anything_3.model.utils.head_utils import custom_interpolate

        if len(feats) < 4:
            raise ValueError(f"Expected 4 feature levels from DA3 backbone, got {len(feats)}")

        tokens0 = feats[0][0]
        if tokens0.ndim != 4:
            raise ValueError(f"Expected tokens shape [Bpair,2,N,C], got {tuple(tokens0.shape)}")

        Bpair, S, N, C = tokens0.shape
        if S != 2:
            # FoundationStereo stereo pair case should be 2; keep generic but fail loudly for now.
            raise ValueError(f"Expected S==2 views for stereo, got S={S}")

        ph, pw = H // self.patch_size, W // self.patch_size
        if ph * pw != N:
            raise ValueError(
                f"Token length N does not match patch grid: N={N}, ph*pw={ph*pw} "
                f"(H={H}, W={W}, patch_size={self.patch_size})."
            )

        # Follow DA3 head logic (see `depth_anything_3/model/dpt.py::DPT._forward_impl`):
        # - flatten [Bpair, 2, N, C] -> [Bpair*2, N, C]
        feats_flat = [f[0].reshape(Bpair * S, N, C) for f in feats]

        # Use head's configured intermediate indices if present; fall back to first 4 levels.
        take_indices = list(getattr(self.head, "intermediate_layer_idx", (0, 1, 2, 3)))
        if len(take_indices) < 4:
            raise ValueError(
                f"Expected head.intermediate_layer_idx to have 4 indices, got {take_indices}"
            )
        if max(take_indices) >= len(feats_flat):
            raise ValueError(
                f"Head expects feature index {max(take_indices)} but backbone returned only {len(feats_flat)} levels"
            )

        resized_feats = []
        patch_start_idx = 0
        for stage_idx, take_idx in enumerate(take_indices[:4]):
            x = feats_flat[take_idx][:, patch_start_idx:]  # [BS, N_patch, C]
            x = self.head.norm(x)  # normalization
            x = x.permute(0, 2, 1).contiguous().reshape(Bpair * S, C, ph, pw)  # [BS, C, ph, pw]

            x = self.head.projects[stage_idx](x)
            if bool(getattr(self.head, "pos_embed", False)):
                x = self.head._add_pos_embed(x, W, H)  # noqa: SLF001 (intentional internal reuse)
            x = self.head.resize_layers[stage_idx](x)
            resized_feats.append(x)

        # Fuse pyramid (DPT returns tensor; DualDPT returns (tensor, aux_list))
        fused = self.head._fuse(resized_feats)  # noqa: SLF001
        if isinstance(fused, (tuple, list)):
            fused = fused[0]

        # Match DA3 head behavior: always apply output_conv1 before interpolating to (h_out, w_out).
        if not (hasattr(self.head, "scratch") and hasattr(self.head.scratch, "output_conv1")):
            raise AttributeError("DA3 head does not expose scratch.output_conv1; cannot extract 128-ch dense feature")
        fused = self.head.scratch.output_conv1(fused)
        if fused.shape[1] != 128:
            raise ValueError(
                f"Expected 128-channel dense feature after output_conv1, got C={fused.shape[1]}"
            )

        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)
        fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)
        if bool(getattr(self.head, "pos_embed", False)):
            fused = self.head._add_pos_embed(fused, W, H)  # noqa: SLF001

        # [BS, 128, H, W] -> [Bpair, 2, 128, H, W]
        fused = fused.view(Bpair, S, fused.shape[1], fused.shape[2], fused.shape[3])
        return fused


