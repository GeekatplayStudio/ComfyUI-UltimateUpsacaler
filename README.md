# Geekatplay Studio HyperTile Upscaler For ComfyUI

This repository packages the Geekatplay Studio tiled upscaling stack for ComfyUI: installer, helper nodes, and ready-to-import workflows for SDXL and FLUX refinement.

## Release Contents

- A one-click installer that clones the required custom nodes and downloads the model stack into the active ComfyUI install.
- HyperTile helper nodes for planning output size, optional Florence captioning, prompt composition, resize staging, tile preview, and regional conditioning.
- A production SDXL upscale workflow in `workflows/hyper_tile_sdxl.json`.
- A FLUX tiled refiner workflow in `workflows/hyper_tile_flux_tiled_refiner.json` plus a separate FLUX refiner profile in `profiles/hyper_tile_flux_refiner_profile.json`.

## Requirements

- A recent ComfyUI build.
- Python 3.10 or newer.
- 12 GB VRAM or more recommended for the bundled defaults.
- Internet access if you want the installer to clone dependencies and download models.

The bundled workflows expect these custom nodes, which the installer will clone automatically:

- `ComfyUI-Manager`
- `ComfyUI_UltimateSDUpscale`
- `Comfyui_TTP_Toolset`

## Install

### One-Click Install

Windows:

```bat
install.bat --comfyui M:\ComfyUI
```

macOS or Linux:

```bash
./install_hyper_tile.sh --comfyui /path/to/ComfyUI
```

The launcher scripts perform a full install by default. That includes the optional FLUX checkpoints and the FLUX Union ControlNet assets.

### Direct Python Install

If you want the base SDXL setup only, call `install.py` directly:

```bash
python install.py --comfyui /path/to/ComfyUI
```

To include the FLUX assets with the direct installer:

```bash
python install.py --comfyui /path/to/ComfyUI --with-flux-assets
```

If you already have the FLUX dev checkpoint locally, you can copy it into the expected filename during install:

```bat
install.bat --comfyui M:\ComfyUI --flux-checkpoint D:\models\flux1-dev-fp8.safetensors
```

`install_hyper_tile.bat` remains in the repo as a compatibility alias and forwards to `install.bat`.

### What The Installer Adds

- Python requirements from `requirements.txt`
- `4x-UltraSharp.pth`
- `RealESRGAN_x4plus.pth`
- `sdxl_base_1.0.safetensors`
- `controlnet_tile_sdxl_1_0.safetensors`
- `flux1-dev-fp8.safetensors` when FLUX assets are enabled
- `flux1-schnell-fp8.safetensors` when FLUX assets are enabled
- `flux_controlnet_union_pro.safetensors` when FLUX assets are enabled

If a remote host blocks direct model download because of license acceptance or authentication rules, the installer prints the exact manual fallback source instead of failing without context.

## Optional Captioning Dependencies

The bundled workflows keep Florence captioning disabled by default. If you install this project with `pip` instead of using `install.py`, the base package stays lean and the captioning stack is exposed as an extra:

```bash
pip install .[captioning]
```

`install.py` still installs the full runtime requirements so the bundled workflows work without additional package management.

## Included Nodes

- `HyperTile Planner`: Detects input size, resolves target size, and recommends tile size, denoise, and tiled decode settings.
- `HyperTile Caption Tiles`: Splits the first image frame into a spatial grid and optionally captions those regions with Florence-2.
- `HyperTile Prompt Composer`: Merges prompt fragments with optional deduplication.
- `HyperTile Resize Image`: Resizes the stage-1 result to the exact planned output size.
- `HyperTile Tile Preview`: Draws the tile layout on a preview image before sampling.
- `HyperTile Regional Conditioning`: Converts tile captions and positions into area-scoped conditioning.

## Workflow Usage

### SDXL Workflow

Import `workflows/hyper_tile_sdxl.json` into ComfyUI.

Recommended flow:

1. Load the source image.
2. Select the SDXL checkpoint.
3. Set the base prompt in `HyperTile Prompt Composer`.
4. Choose the planner sizing mode: `target_long_edge`, `magnification`, or `output_size`.
5. Confirm the tile overlay from `HyperTile Tile Preview`.
6. Adjust `vram_gb` or `denoise_adjust` only if you need to trade detail for seam safety.
7. Queue the graph.

The bundled graph automatically preserves aspect ratio, resizes the stage-1 upscale to the planned output size, and keeps the tiled denoise pass aligned with the planner outputs.

The installer also downloads `sdxl_base_1.0.safetensors` so the packaged SDXL workflow has a matching default checkpoint available in a fresh ComfyUI setup.

### FLUX Workflow

Import `workflows/hyper_tile_flux_tiled_refiner.json` when you want a tiled FLUX pass instead of only the profile file.

The FLUX workflow:

1. Detects source size and plans the final output.
2. Performs the neural upscale pass.
3. Resizes to the exact planned output size.
4. Shows a tile preview.
5. Splits the image into tiles with TTP nodes.
6. Optionally captions tiles with Florence-2.
7. Loads the FLUX checkpoint and FLUX Union ControlNet.
8. Samples from VAE-encoded tiles and reassembles the final image.

Important limits:

- It requires a recent ComfyUI build with the FLUX and SD3-era core nodes.
- It is designed around `flux1-dev-fp8.safetensors`, and `flux1-schnell-fp8.safetensors` is also available for faster passes.
- It expects `flux_controlnet_union_pro.safetensors` in `models/controlnet`.
- The default graph keeps one merged positive prompt across the tile batch. `HyperTile Regional Conditioning` is shipped for advanced graphs, but it is not wired into the default FLUX workflow.

## FLUX Final Pass Profile

`profiles/hyper_tile_flux_refiner_profile.json` is a settings profile, not a ComfyUI workflow graph. Use it as a reference after finishing the SDXL pass or when building your own FLUX refinement graph.

## Help

- Use the SDXL workflow for the first major upscale jump.
- Use the FLUX workflow only after the image is already resized and composition-locked.
- Keep tile size at `512` on 12-16 GB GPUs unless you have validated a larger tile budget locally.
- Florence-2 will populate its cache on first use, so the first caption run is slower than later runs.

## Operational Notes

- `HyperTile Caption Tiles` downloads Florence-2 into the active Python cache the first time you run it, but the bundled workflows ship with captioning disabled by default.
- Hugging Face metadata warnings during Florence setup are expected and are not core ComfyUI model-loading errors.
- The bundled workflow targets standard image save nodes. Alternate output formats may require extra ComfyUI extensions.
- The bundled FLUX assets are optional because they are large and many users only need the SDXL production path.
- Third-party nodes, checkpoints, and models retain their own licenses and terms.

See `RELEASE_NOTES.md` for the cleanup and release-hardening summary.

## License

This repository's source code is released under the MIT License. See `LICENSE`.