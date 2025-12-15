# Running `scripts/mesh_from_mask.py`

This script wraps the official SAM-3D inference notebook so you can turn an RGB image and a binary object mask into Gaussian splats, a textured mesh (`.glb`), and a quick preview render.

## 1. Environment and checkpoints

1. Activate the prepared conda env (or follow `doc/setup.md` to create it):
   ```bash
   conda activate sam3d-objects
   ```
2. Download the released weights once with the Hugging Face CLI:
   ```bash
   cd /home/rahman79/Desktop/Projects/sam-3d-objects
   TAG=hf
   hf download --repo-type model --local-dir checkpoints/${TAG}-download --max-workers 1 facebook/sam-3d-objects
   rm -rf checkpoints/hf
   mv checkpoints/${TAG}-download checkpoints/hf
   cp checkpoints/hf/checkpoints/* checkpoints/hf/
   cp checkpoints/hf/checkpoints/pipeline.yaml checkpoints/hf/pipeline.yaml
   ```

## 2. Grab a real image + mask

Below is one example using the `FudanPed00054` pedestrian sample provided by the torchvision project (license compatible with research use):
```bash
mkdir -p notebook/images/fudan_ped
curl -L -o notebook/images/fudan_ped/image.png \
  https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/FudanPed00054.png
curl -L -o notebook/images/fudan_ped/mask.png \
  https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/FudanPed00054_mask.png
```
The mask file is a true binary silhouette supplied by the dataset, so no synthetic mask editing is required.

## 3. Run the mesh generator

```bash
python scripts/mesh_from_mask.py \
  --config checkpoints/hf/pipeline.yaml \
  --image notebook/images/fudan_ped/image.png \
  --mask notebook/images/fudan_ped/mask.png \
  --output-dir artifacts/mesh_demo_fudan \
  --seed 42
```

Outputs inside `artifacts/mesh_demo_fudan/`:
- `scene_gaussians.ply`: Gaussian splats reconstructed from the scene.
- `scene.glb`: Textured mesh that can be opened in Blender or the notebook Plotly widget.
- `scene_preview.png`: Quick look render generated with trimesh/pyglet in headless mode.

### Tips
- Pass `--compile` to enable `torch.compile` for the backbone after the first warm-up run (reduces sampling time on subsequent calls).
- Use `--preview-res` to trade rendering fidelity for speed in the PNG preview.
- The script expects a single connected mask; if your dataset exposes multi-class labels (e.g., ADE20K), extract one object ID into a binary PNG before calling the script.
