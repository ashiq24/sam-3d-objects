# SAM 3D Objects – Repository Notes

## High-Level Layout
- `README.md`, `demo.py`, and `plan.md` describe usage, demos, and roadmap.
- `checkpoints/` stores released Hydra configs (`pipeline.yaml`) and weight folders (`*.ckpt`, `*.safetensors`). Use these paths when instantiating the inference pipeline.
- `doc/setup.md` plus the various `requirements*.txt` files capture installation profiles (`dev`, `inference`, `p3d`). `environments/default.yml` pins the CUDA 12.1 conda base used here.
- `notebook/` contains the public API (`inference.py`), three runnable demos, helper meshes/gaussians, and sample images/masks.
- `patching/` and `hydra/` hold override configs for experiments. `images/`, `meshes/`, and `patching/` assets plug directly into the demos.
- `sam3d_objects/` is the main package. Everything described below lives inside this module.

## Package Map (`sam3d_objects/`)
- `config/` – Hydra config utilities and defaults used to build the end-to-end pipeline.
- `data/` – dataset loaders and preprocessing code. The `dataset/tdfy/` sub-tree includes mask/image transforms (`img_and_mask_transforms.py`), image processing helpers, pose target conversion, and 3D-specific augmentations.
- `model/`
	- `io.py` centralizes checkpoint loading, prefix filtering, sharded weight handling, and utility functions such as `filter_and_remove_prefix_state_dict_fn()`.
	- `backbone/` contains the learnable components. Within it:
		- `generator/` implements flow-matching backbones (`base.py`, `classifier_free_guidance.py`, `flow_matching/`).
		- `tdfy_dit/` provides DiT-style latent models, sparse tensor layers, attention blocks, and decoders (see `models/*.py`).
		- `layers/llama3/` currently ships the LLaMA3 FFN implementation used for text-conditioning experiments.
- `pipeline/` orchestrates inference and preprocessing. Key files are `inference_pipeline.py`, `inference_pipeline_pointmap.py`, `inference_utils.py`, `layout_post_optimization_utils.py`, and `preprocess_utils.py`.
- `pipeline/depth_models/` wraps auxiliary models such as MoGe for pointmaps.
- `utils/visualization/` exposes helpers to view gaussians/meshes via matplotlib, plotly, or the custom `SceneVisualizer`.

## Inference Flow
1. **Public API (`notebook/inference.Inference`)** – loads a Hydra config, enforces a whitelist for instantiation safety, merges an object mask into the alpha channel, and delegates to `InferencePipelinePointMap`.
2. **Preprocessing** – `InferencePipeline.preprocess_image()` applies the joint RGB/mask transforms defined in the Hydra config (`preprocess_utils`). Masks are always embedded into the alpha channel of the RGBA tensor.
3. **Stage 1 – Sparse Structure** – `InferencePipeline.sample_sparse_structure()` (see [sam3d_objects/pipeline/inference_pipeline.py](sam3d_objects/pipeline/inference_pipeline.py#L176-L302)) uses the stage-1 generator to synthesize a coarse occupancy grid. The output voxels (`coords`) are optionally pruned via `prune_sparse_structure()` and downsampled before being passed downstream. Pose is decoded with `get_pose_decoder()` from [sam3d_objects/pipeline/inference_utils.py](sam3d_objects/pipeline/inference_utils.py#L189-L287).
4. **Stage 2 – Structured Latent (SLAT)** – `sample_slat()` conditions on the input image and sparse coordinates to produce a structured latent (`sp.SparseTensor`). The latent is denormalized via the stored mean/std vectors.
5. **Decoding** – `decode_slat()` chooses the requested format(s): gaussians (`slat_decoder_gs`, optional 4k variant) and/or mesh (`slat_decoder_mesh`).
6. **Post-processing** – `postprocess_slat_output()` calls `postprocessing_utils.to_glb()` to simplify the mesh, bake textures, and write GLB/PLY files. Layout refinement lives in `layout_post_optimization_utils.py` and is triggered from notebooks when needed.

## Main Class & Module Structure

### Public Entry Classes
- **`Inference`** – High-level wrapper that:
	- Loads configs via OmegaConf and instantiates `InferencePipelinePointMap`.
	- Provides utility functions for merging masks, loading images, rendering gaussian turntables (`render_video()`), and constructing multi-object scenes (see [notebook/inference.py](notebook/inference.py)).
- **`InferencePipeline` / `InferencePipelinePointMap`** – Core orchestrators. They materialize all sub-models, preprocessors, and condition embedders. The `run()` method wires the full flow (preprocess → stage1 → pose decode → stage2 → decode → postprocess) and surfaces Gaussian / mesh / GLB artifacts.

### Stage 1 – Sparse Structure Generator
- **`FlowMatching`** ([sam3d_objects/model/backbone/generator/flow_matching/model.py](sam3d_objects/model/backbone/generator/flow_matching/model.py))
	- Base sampler that implements training (`loss()`), inference (`generate_iter()`), and log-likelihood estimation.
	- Accepts a `reverse_fn` (typically wrapped in CFG) plus solver choices (`Euler`, `Midpoint`, `RK4`, `SDE`).
	- Supports conditional sampling through `_generate_dynamics()` and Hutchinson divergence estimates for likelihood evaluation.
- **`ClassifierFreeGuidance` / `PointmapCFG`** ([sam3d_objects/model/backbone/generator/classifier_free_guidance.py](sam3d_objects/model/backbone/generator/classifier_free_guidance.py))
	- Wraps any generator to mix conditional/unconditional predictions. The SAM3D configs set `unconditional_handling="add_flag"` so the backbone can zero specific modalities (e.g., drop pointmaps during CFG).
	- `PointmapCFG` extends CFG to tri-modally mix masked pointmaps for the MoGe conditioning route.
- **`SparseStructureFlowModel` + `SparseStructureFlowTdfyWrapper`** ([sam3d_objects/model/backbone/tdfy_dit/models/sparse_structure_flow.py](sam3d_objects/model/backbone/tdfy_dit/models/sparse_structure_flow.py))
	- DiT-style transformer over voxel patches with timestep embeddings and optional pose tokens.
	- The wrapper reshapes sparse tensors, injects condition embeddings, and returns the flattened latent consumed by the decoder.

### Stage 2 – Structured Latent Decoder
- **`SLatFlowModel` + `SLatFlowModelTdfyWrapper`** ([sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_flow.py](sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_flow.py))
	- Operates directly on sparse tensors (`sp.SparseTensor`).
	- Uses a U-Net-like stack of sparse residual blocks plus transformer cross-attention for conditioning on image tokens / pointmaps.
	- Optional shortcut/dilation embeddings (`d_embedder`) allow distillation or stage-2 CFG tweaks.
- **Decoders** – `slat_decoder_mesh`, `slat_decoder_gs`, and the optional 4k Gaussian decoder live under `tdfy_dit` modules; they map structured latents to explicit meshes or Gaussian splats.

### Supporting Modules
- **`model/io.py`** – Shared checkpoint utilities: suffix-based weight remapping, prefix stripping, sharded checkpoint loading, and state-dict munging used throughout pipeline initialization.
- **`pipeline/inference_utils.py`** – Pose decoding utilities (quaternion/scale reconstruction, layout optimization via ICP + render-and-compare), SLAT normalization stats, and helper functions like `layout_post_optimization()`.
- **`pipeline/preprocess_utils.py`** – Factory functions for image/mask preprocessing pipelines injected via Hydra configs.
- **Visualization (`utils/visualization`)** – `SceneVisualizer` and Plotly/Matplotlib helpers to render meshes, Gaussians, and layout diagnostics.

## Operational Notes
- **Configs and Seeds** – Hydra configs specify every checkpoint/config path relative to the `workspace_dir` you pass into `InferencePipeline`. When running custom jobs, copy the desired `checkpoints/<tag>/pipeline.yaml` and override paths as needed.
- **Condition Embedders** – Both stage-1 and stage-2 generators lazily load their `condition_embedder` submodules once, caching them in `self.condition_embedders` to avoid repeated instantiation.
- **Precision / Compilation** – `dtype` defaults to bfloat16. Setting `compile_model=True` compiles the embedding, generator, and decoder forward passes with `torch.compile` (max-autotune) and performs a warmup run.
- **Post-processing** – `postprocessing_utils.to_glb()` handles mesh simplification (quadric error), baking (texture atlases), and `GLB` export; `outputs["gs"]` and `outputs["mesh"]` remain in-memory `gaussian-splat`/PyTorch3D objects for downstream tooling.

This document should give enough context to navigate the repository, trace how Hydra configs assemble the two-stage pipeline, and understand where to modify or extend the main models.
