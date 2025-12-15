#!/usr/bin/env python3
"""Simple script to generate a SAM 3D mesh from an image + mask and save quick-look renders."""
# python scripts/mesh_from_mask.py --image notebook/images/kid_box/image.png --mask notebook/images/kid_box/0.png --output-dir artifacts/kid_box
from __future__ import annotations

import argparse
import importlib
import os
import sys
import types
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYGLET_HEADLESS", "true")
os.environ.setdefault("PYGLET_SHADOW_WINDOW", "1")

import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebook"
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.append(str(NOTEBOOK_DIR))


def _ensure_kaolin_stub() -> None:
    """Provide a lightweight kaolin stub if the real package is unavailable."""

    try:
        importlib.import_module("kaolin")
        return
    except ModuleNotFoundError:
        pass

    class _Placeholder:  # pylint: disable=too-few-public-methods
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    kaolin_mod = types.ModuleType("kaolin")

    visualize_mod = types.ModuleType("kaolin.visualize")

    class _Visualizer(_Placeholder):
        def show(self, *args, **kwargs):
            raise RuntimeError(
                "Kaolin visualization is unavailable in this environment."
            )

    visualize_mod.IpyTurntableVisualizer = _Visualizer

    render_mod = types.ModuleType("kaolin.render")
    camera_mod = types.ModuleType("kaolin.render.camera")

    class _CameraBase(_Placeholder):
        pass

    camera_mod.Camera = _CameraBase
    camera_mod.CameraExtrinsics = _CameraBase
    camera_mod.PinholeIntrinsics = _CameraBase

    render_mod.camera = camera_mod

    utils_mod = types.ModuleType("kaolin.utils")
    testing_mod = types.ModuleType("kaolin.utils.testing")

    def check_tensor(*args, **kwargs):  # pylint: disable=unused-argument
        return True

    testing_mod.check_tensor = check_tensor
    utils_mod.testing = testing_mod

    kaolin_mod.visualize = visualize_mod
    kaolin_mod.render = render_mod
    kaolin_mod.utils = utils_mod

    sys.modules["kaolin"] = kaolin_mod
    sys.modules["kaolin.visualize"] = visualize_mod
    sys.modules["kaolin.render"] = render_mod
    sys.modules["kaolin.render.camera"] = camera_mod
    sys.modules["kaolin.utils"] = utils_mod
    sys.modules["kaolin.utils.testing"] = testing_mod


_ensure_kaolin_stub()

from inference import (  # type: ignore  # imported via notebook path
    Inference,
    load_image,
    load_mask,
)


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _resolve_relative_to_repo(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def save_preview(mesh: trimesh.Trimesh, output_path: Path, resolution: int) -> Optional[Path]:
    scene = trimesh.Scene(mesh)
    try:
        # Pyglet offscreen rendering works only in headless mode when visible=False.
        png_bytes = scene.save_image(resolution=(resolution, resolution), visible=False)
    except BaseException as exc:  # pragma: no cover - best-effort visualization
        print(f"[WARN] Could not render preview: {exc}")
        return None
    output_path.write_bytes(png_bytes)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="checkpoints/hf/pipeline.yaml",
        help="Relative or absolute path to the Hydra pipeline config.",
    )
    parser.add_argument("--image", required=True, help="Path to the RGB image.")
    parser.add_argument(
        "--mask",
        required=True,
        help="Single-object binary mask (same spatial size as the image).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/mesh_demo",
        help="Directory for meshes, splats, and preview renders (relative to repo root).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--preview-res",
        type=int,
        default=768,
        help="Resolution of the saved mesh preview PNG.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile on the inference pipeline (longer warmup, faster sampling).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = _resolve_relative_to_repo(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    image_path = _resolve(args.image)
    mask_path = _resolve(args.mask)
    if not image_path.exists() or not mask_path.exists():
        raise FileNotFoundError("Image or mask path is invalid.")

    output_dir = _resolve_relative_to_repo(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading pipeline from {config_path}")
    inference = Inference(str(config_path), compile=args.compile)

    print(f"[INFO] Running inference (seed={args.seed})")
    image = load_image(str(image_path))
    mask = load_mask(str(mask_path))
    with torch.inference_mode():
        result = inference(image, mask, seed=args.seed)

    # Export gaussian splats
    gaussian = result.get("gs")
    if gaussian is not None:
        ply_path = output_dir / "scene_gaussians.ply"
        gaussian.save_ply(str(ply_path))
        print(f"[OK] Saved Gaussian splats to {ply_path}")

    # Export mesh / GLB
    mesh_obj = result.get("glb")
    if mesh_obj is None:
        print("[WARN] Pipeline did not return a mesh. Check mask coverage and try again.")
        return

    glb_path = output_dir / "scene.glb"
    mesh_obj.export(glb_path)
    print(f"[OK] Saved mesh to {glb_path}")

    preview_path = output_dir / "scene_preview.png"
    saved_preview = save_preview(mesh_obj, preview_path, resolution=args.preview_res)
    if saved_preview:
        print(f"[OK] Saved preview render to {saved_preview}")


if __name__ == "__main__":
    main()
