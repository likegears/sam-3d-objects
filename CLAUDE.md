# CLAUDE.md - AI Assistant Guide for SAM 3D Objects

This document provides comprehensive guidance for AI assistants working with the SAM 3D Objects codebase.

## Project Overview

**SAM 3D Objects** is a foundation model for reconstructing full 3D shape geometry, texture, and layout from single images. It excels in real-world scenarios with occlusion and clutter, using progressive training and a data engine with human feedback.

- **Paper**: [SAM 3D: 3Dfy Anything in Images](https://ai.meta.com/research/publications/sam-3d-3dfy-anything-in-images/)
- **Website**: https://ai.meta.com/sam3d/
- **Demo**: https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d
- **Related**: [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) for human mesh recovery

### Key Capabilities

- Single-image to 3D reconstruction
- Multi-object scene reconstruction
- Mask-based object segmentation
- Output formats: Gaussian Splats (.ply), Meshes, GLB files
- Handles occlusions, unusual poses, and challenging natural scenes

## Repository Structure

```
sam-3d-objects/
├── sam3d_objects/              # Main package
│   ├── pipeline/              # Inference pipelines
│   │   ├── inference_pipeline.py          # Core inference pipeline
│   │   ├── inference_pipeline_pointmap.py # Extended pipeline with pointmap
│   │   ├── depth_models/                  # Depth estimation models (MoGe)
│   │   └── preprocess_utils.py            # Preprocessing utilities
│   ├── model/                 # Model architectures
│   │   ├── backbone/          # Core model components
│   │   │   ├── dit/           # Diffusion Transformer embedders
│   │   │   │   └── embedder/  # DINO, pointmap embedders
│   │   │   ├── generator/     # Generation models
│   │   │   │   ├── flow_matching/  # Flow matching solver
│   │   │   │   └── shortcut/       # Distillation shortcuts
│   │   │   └── tdfy_dit/      # Main 3Dfy DiT architecture
│   │   │       ├── models/    # VAE encoders/decoders
│   │   │       │   └── structured_latent_vae/
│   │   │       │       ├── encoder.py      # Sparse structure encoder
│   │   │       │       ├── decoder_gs.py   # Gaussian splat decoder
│   │   │       │       ├── decoder_mesh.py # Mesh decoder
│   │   │       │       └── decoder_rf.py   # Radiance field decoder
│   │   │       ├── modules/   # Transformer & attention modules
│   │   │       │   ├── sparse/            # Sparse tensor operations
│   │   │       │   └── transformer/       # Modulated transformers
│   │   │       ├── renderers/ # Gaussian & octree renderers
│   │   │       ├── representations/ # 3D representations
│   │   │       │   ├── gaussian/    # Gaussian splatting
│   │   │       │   ├── mesh/        # Mesh (FlexiCubes)
│   │   │       │   ├── octree/      # Octree structures
│   │   │       │   └── radiance_field/  # NeRF-style fields
│   │   │       └── utils/     # Rendering & postprocessing
│   │   ├── layers/            # Custom layers (LLaMA3 FF)
│   │   └── io.py              # Model I/O utilities
│   ├── data/                  # Data processing
│   │   └── dataset/tdfy/      # 3Dfy dataset processing
│   │       ├── preprocessor.py           # Data preprocessor
│   │       ├── img_processing.py         # Image transformations
│   │       ├── img_and_mask_transforms.py # Joint transforms
│   │       ├── pose_target.py            # Pose conventions
│   │       └── transforms_3d.py          # 3D transformations
│   ├── config/                # Configuration utilities
│   └── utils/                 # General utilities
│       └── visualization/     # Scene visualization tools
├── notebook/                  # Examples & inference
│   ├── inference.py           # Public-facing API
│   ├── mesh_alignment.py      # SAM 3D Body alignment
│   ├── demo_single_object.ipynb    # Single object tutorial
│   ├── demo_multi_object.ipynb     # Multi-object tutorial
│   ├── demo_3db_mesh_alignment.ipynb # Human mesh alignment
│   └── images/                # Example images & masks
├── demo.py                    # Quick start demo script
├── environments/              # Conda environment specs
│   └── default.yml           # Default environment
├── patching/                  # Patches for dependencies
│   └── hydra/                # Hydra patches
├── doc/                       # Documentation
│   └── setup.md              # Setup instructions
├── pyproject.toml            # Package configuration
├── requirements.txt          # Core dependencies
├── requirements.p3d.txt      # PyTorch3D dependencies
├── requirements.inference.txt # Inference-specific deps
└── requirements.dev.txt      # Development dependencies
```

## Core Architecture

### Two-Stage Pipeline

SAM 3D Objects uses a hierarchical generation approach:

#### Stage 1: Sparse Structure Generation
- **Input**: RGBA image (RGB + alpha mask)
- **Model**: `ss_generator` (Sparse Structure Generator)
- **Embedder**: DINO-based condition embedder
- **Output**: 3D voxel occupancy grid (sparse coordinates)
- **Decoder**: `ss_decoder` converts latent to binary occupancy
- **Configuration**: CFG strength=7, inference steps=25, rescale_t=3

#### Stage 2: Structured Latent Generation
- **Input**: Image + sparse coordinates from Stage 1
- **Model**: `slat_generator` (Structured Latent Generator)
- **Embedder**: Pointmap or DINO embedder
- **Output**: Sparse latent tensor (per-voxel features)
- **Configuration**: CFG strength=5, inference steps=25, rescale_t=3

#### Stage 3: Decoding to 3D Representations
- **Gaussian Decoder** (`slat_decoder_gs`): Generates 3D Gaussian Splats
- **Mesh Decoder** (`slat_decoder_mesh`): Generates triangle meshes via FlexiCubes
- **Radiance Field Decoder** (optional): NeRF-style representations

### Key Components

1. **Flow Matching**: Used for generative modeling instead of traditional diffusion
2. **Classifier-Free Guidance (CFG)**: Strength and interval control for generation quality
3. **Sparse Tensors**: Efficient representation using spconv library
4. **Modulated Transformers**: DiT-style architecture with adaptive layer norm
5. **Multi-Modal Conditioning**: DINO features + pointmaps for layout control

## Development Workflows

### Environment Setup

```bash
# Create environment (requires mamba or conda)
mamba env create -f environments/default.yml
mamba activate sam3d-objects

# Install dependencies (requires CUDA 12.1)
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'  # PyTorch3D (install separately)

# Install inference dependencies
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

# Apply patches
./patching/hydra  # Hydra PR #2863
```

### Getting Checkpoints

Checkpoints are hosted on HuggingFace and require authentication:

```bash
pip install 'huggingface-hub[cli]<1.0'
huggingface-cli login  # Use access token

TAG=hf
hf download --repo-type model --local-dir checkpoints/${TAG}-download \
  --max-workers 1 facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```

**Important**: Access must be requested on [HuggingFace](https://huggingface.co/facebook/sam-3d-objects). Sanctioned jurisdictions will be rejected.

### Hardware Requirements

- **GPU**: NVIDIA GPU with **at least 32GB VRAM** (A100, H100, H200 recommended)
- **Platform**: Linux 64-bit architecture (`linux-64`)
- **CUDA**: Version 12.1
- **Build Node**: Some packages (PyTorch3D) require building on compute nodes with GPU

### Running Inference

#### Quick Start (demo.py)

```python
import sys
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# Load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# Load image and mask
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# Run model
output = inference(image, mask, seed=42)

# Export results
output["gs"].save_ply("splat.ply")  # Gaussian splat
# output["glb"]  # GLB mesh with texture
```

#### Advanced Options

```python
output = inference._pipeline.run(
    image,
    mask,
    seed=42,
    stage1_only=False,              # Set True to only get sparse structure
    with_mesh_postprocess=True,     # Enable mesh simplification
    with_texture_baking=True,       # Bake texture into mesh
    use_vertex_color=False,         # Use vertex colors vs texture map
    stage1_inference_steps=25,      # Override default steps
    stage2_inference_steps=25,
    use_stage1_distillation=False,  # Use distilled shortcuts
    use_stage2_distillation=False,
    decode_formats=["gaussian", "mesh"],  # Output formats
)
```

#### Multi-Object Scenes

```python
from inference import make_scene, render_video

outputs = []
for mask_idx in range(num_objects):
    mask = load_single_mask(image_folder, index=mask_idx)
    output = inference(image, mask, seed=42)
    outputs.append(output)

# Merge into scene
scene_gs = make_scene(*outputs)

# Render turntable video
frames = render_video(scene_gs, resolution=512, num_frames=300)
```

### Configuration Management

SAM 3D Objects uses **Hydra** for configuration. Key files:
- `checkpoints/hf/pipeline.yaml` - Main pipeline config
- Individual model configs referenced within pipeline config

**Safety**: The codebase implements whitelist/blacklist filters for Hydra instantiation to prevent unsafe operations (see `notebook/inference.py:315-342`).

### Output Formats

1. **Gaussian Splat** (`.ply`): Efficient 3D representation for real-time rendering
2. **Mesh** (`.glb`): Triangle mesh with optional texture baking
3. **Sparse Structure**: Voxel occupancy (Stage 1 output)
4. **Structured Latent**: Per-voxel features (Stage 2 output)

**Key Output Fields**:
- `output["gs"]` - Gaussian splat model object
- `output["glb"]` - GLB file (trimesh object)
- `output["mesh"]` - Raw mesh before postprocessing
- `output["coords"]` - Sparse voxel coordinates
- `output["rotation"]`, `output["translation"]`, `output["scale"]` - Pose in scene frame

## Code Conventions & Best Practices

### Coding Style

From `CONTRIBUTING.md`:
- **Indentation**: 2 spaces (not tabs)
- **Line Length**: 80 characters
- **Linting**: Code must pass linting checks
- **Testing**: Add tests for new functionality
- **Documentation**: Update docs for API changes

### File Headers

All source files include:
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
```

### Import Conventions

```python
# Standard library
import os
import sys

# Third-party
import torch
import numpy as np
from PIL import Image

# Project imports
from sam3d_objects.pipeline import preprocess_utils
from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
```

### Configuration Patterns

Hydra configs use `_target_` for class instantiation:

```yaml
module:
  _target_: sam3d_objects.model.SomeModel
  param1: value1
  param2:
    _target_: sam3d_objects.utils.SomeUtil
```

### Model Loading Pattern

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

config = OmegaConf.load("path/to/config.yaml")
model = instantiate(config)
```

### Sparse Tensor Operations

The codebase uses custom sparse tensor classes (`sam3d_objects.model.backbone.tdfy_dit.modules.sparse`):

```python
import sam3d_objects.model.backbone.tdfy_dit.modules.sparse as sp

# Create sparse tensor
sparse_tensor = sp.SparseTensor(
    coords=coords,  # [N, 4] tensor: [batch_idx, x, y, z]
    feats=features   # [N, C] tensor: features per point
)
```

### Attention Backend Selection

Automatically selects optimal backend based on GPU:

```python
# In inference_pipeline.py:11-21
if "A100" in gpu_name or "H100" in gpu_name or "H200" in gpu_name:
    os.environ["ATTN_BACKEND"] = "flash_attn"
    os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"
```

### Preprocessing Pipeline

Images must be RGBA format with mask in alpha channel:

```python
# From inference.py:94-99
def merge_mask_to_rgba(self, image, mask):
    mask = mask.astype(np.uint8) * 255
    mask = mask[..., None]
    rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
    return rgba_image
```

### Model Compilation (Optional)

For faster inference, models support torch.compile:

```python
inference = Inference(config_path, compile=True)
# Triggers warmup with 3 iterations
```

**Note**: Compilation increases initialization time but speeds up inference.

## Key Files for AI Assistants

### Entry Points

1. **`demo.py`** (23 lines): Simplest usage example
2. **`notebook/inference.py`** (415 lines): Public API with safety checks
3. **`sam3d_objects/pipeline/inference_pipeline.py`** (845 lines): Core pipeline implementation
4. **`sam3d_objects/pipeline/inference_pipeline_pointmap.py`**: Extended pipeline with layout control

### Model Architectures

1. **`sam3d_objects/model/backbone/generator/flow_matching/model.py`**: Flow matching generator
2. **`sam3d_objects/model/backbone/tdfy_dit/models/sparse_structure_flow.py`**: Stage 1 model
3. **`sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_flow.py`**: Stage 2 model
4. **`sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_vae/`**: Encoders/decoders

### Data Processing

1. **`sam3d_objects/data/dataset/tdfy/preprocessor.py`**: Data preprocessing
2. **`sam3d_objects/data/dataset/tdfy/img_and_mask_transforms.py`**: Image & mask transforms
3. **`sam3d_objects/pipeline/preprocess_utils.py`**: Preprocessing utilities

### Utilities

1. **`sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py`**: Mesh postprocessing, texture baking
2. **`sam3d_objects/model/backbone/tdfy_dit/utils/render_utils.py`**: Rendering utilities
3. **`sam3d_objects/utils/visualization/`**: Visualization tools

## Common Tasks for AI Assistants

### Task: Debug Inference Issues

1. **Check GPU memory**: Requires 32GB+ VRAM
2. **Verify CUDA version**: Must be 12.1
3. **Check environment variables**: `ATTN_BACKEND`, `CUDA_HOME`, `LIDRA_SKIP_INIT`
4. **Validate input format**: RGBA uint8 numpy array or PIL Image
5. **Check mask**: Binary mask or alpha channel, same size as image
6. **Review logs**: Uses `loguru` for logging

### Task: Add New Decoder

1. Create decoder in `sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_vae/`
2. Implement `forward(self, slat: sp.SparseTensor) -> representation`
3. Add config file for Hydra instantiation
4. Update `InferencePipeline.init_*_decoder()` method
5. Add to `decode_formats` in pipeline config
6. Update `decode_slat()` method with new format

### Task: Modify Preprocessing

1. Edit `sam3d_objects/data/dataset/tdfy/preprocessor.py`
2. Or create custom preprocessor and pass to pipeline:
   ```python
   from sam3d_objects.pipeline import preprocess_utils

   custom_preprocessor = preprocess_utils.get_default_preprocessor()
   # Modify transforms...

   pipeline = InferencePipeline(
       ss_preprocessor=custom_preprocessor,
       ...
   )
   ```

### Task: Adjust Generation Quality

Modify CFG and inference parameters:

```python
pipeline = InferencePipeline(
    ss_cfg_strength=7,        # Higher = stronger conditioning
    ss_inference_steps=25,    # More steps = better quality
    ss_rescale_t=3,           # Timestep rescaling
    slat_cfg_strength=5,
    slat_inference_steps=25,
    ...
)
```

### Task: Export to Different Format

Outputs can be converted:

```python
# Gaussian Splat to PLY
output["gs"].save_ply("output.ply")

# Mesh to various formats
import trimesh
mesh = output["glb"]
mesh.export("output.obj")
mesh.export("output.stl")
mesh.export("output.glb")
```

### Task: Visualize Intermediate Results

```python
# Stage 1 only (sparse structure)
output = pipeline.run(image, mask, stage1_only=True)
coords = output["coords"]  # [N, 4] tensor: batch_idx, x, y, z
voxels = output["voxel"]   # Normalized coordinates [-0.5, 0.5]

# Visualize sparse structure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(voxels[:, 0], voxels[:, 1], voxels[:, 2])
plt.show()
```

## Important Notes for AI Assistants

### Environment Variables

- **`CUDA_HOME`**: Set to `$CONDA_PREFIX` for compilation
- **`LIDRA_SKIP_INIT`**: Set to `"true"` to skip package initialization (for lightweight tools)
- **`ATTN_BACKEND`**: Auto-set based on GPU (flash_attn for A100/H100/H200)
- **`PIP_EXTRA_INDEX_URL`**: Required for PyTorch/CUDA dependencies

### Common Pitfalls

1. **PyTorch3D Build Issues**: Must build on GPU-enabled node
2. **VRAM Overflow**: Reduce batch size or use smaller images
3. **Hydra Safety**: Don't modify whitelist/blacklist filters without review
4. **Mask Format**: Must be same resolution as image, binary or 0-255
5. **Checkpoint Access**: Requires HuggingFace authentication and approval

### Testing Changes

1. Run `demo.py` as smoke test
2. Test with example images in `notebook/images/`
3. Check both single and multi-object scenarios
4. Verify output formats (Gaussian, mesh, GLB)
5. Test with different seeds for reproducibility

### Performance Optimization

1. **Model Compilation**: Set `compile=True` in Inference (increases startup time)
2. **Distillation**: Use `use_stage1_distillation=True` for faster inference (lower quality)
3. **Inference Steps**: Reduce to 10-15 for faster results
4. **Image Resolution**: Preprocessing resizes to 512x512 by default
5. **Downsample SS**: Increase `downsample_ss_dist` to reduce voxel count

### Related Repositories

- **SAM 3D Body**: https://github.com/facebookresearch/sam-3d-body
- **MoGe (Depth)**: https://github.com/microsoft/MoGe
- **Hydra**: https://hydra.cc/

## License & Contributing

- **License**: SAM License (see `LICENSE` file)
- **Contributing**: See `CONTRIBUTING.md`
- **Code of Conduct**: See `CODE_OF_CONDUCT.md`
- **CLA Required**: Contributors must sign Meta's CLA

## Troubleshooting

### Import Errors

```python
# If you see "No module named 'sam3d_objects'"
pip install -e .

# If you see CUDA errors
export CUDA_HOME=$CONDA_PREFIX
```

### GPU Detection Issues

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Check GPU model
```

### Checkpoint Loading Failures

```bash
# Verify checkpoint structure
ls checkpoints/hf/
# Should contain: pipeline.yaml, *.safetensors or *.ckpt files

# Check HuggingFace auth
huggingface-cli whoami
```

### Out of Memory

- Reduce image resolution
- Use fewer inference steps
- Enable distillation mode
- Close other GPU processes

---

**Last Updated**: 2025-11-21
**Model Version**: SAM 3D Objects v1 (2025-11-19 release)
**Python**: 3.11.0
**PyTorch**: 2.5.1
**CUDA**: 12.1
