# ComfyUI HyperTile Upscaler

This repository is a ComfyUI custom-node package plus installer bundle for large-format generative upscaling.

Repository target: `GeekatplayStudio/ComfyUI-UltimateUpsacaler`

It ships three things:

- A single-click installer that clones the missing custom nodes and downloads the core upscaler and controlnet assets.
- Custom helper nodes for tile planning and caption-guided prompt construction.
- Ready-to-import SDXL and FLUX workflows for the resize, tiled denoise, and final refinement stages.

## What It Builds

The default flow is:

1. Stage 1 neural resize with `4x-UltraSharp` or `RealESRGAN_x4plus`.
2. Stage 2 tiled img2img refinement with `UltimateSDUpscaleNoUpscale`.
3. Optional Stage 3 FLUX refinement using the profile in `workflows/hyper_tile_flux_refiner_profile.json`.

The tiled stage is set up around low denoise, Gaussian-style edge blending through mask blur and padding, and tiled decode to stay stable on 12-16 GB GPUs.

## Included Nodes

- `HyperTile Planner`
  - Recommends tile size, denoise, and tiled decode settings from the source image, target size, model family, and VRAM.
- `HyperTile Caption Tiles`
  - Uses Florence-2 through `transformers` to caption the input image or tile batch and extract texture-aware prompt cues.
- `HyperTile Prompt Composer`
  - Merges the user prompt, tile caption prompt, and a quality suffix into a single deduplicated prompt string.
- `HyperTile Regional Conditioning`
  - Converts TTP tile positions plus caption lines into area-weighted conditioning blocks for spatial prompt control.

## Single-Click Install

On macOS, double-click `install_hyper_tile.command`.

From a terminal:

```bash
./install_hyper_tile.sh --comfyui /path/to/ComfyUI
```

Optional FLUX controlnet download:

```bash
./install_hyper_tile.sh --comfyui /path/to/ComfyUI --with-flux-assets
```

The installer will:

- install this package's Python requirements
- clone `ComfyUI-Manager`, `ComfyUI_UltimateSDUpscale`, and `Comfyui_TTP_Toolset` into `ComfyUI/custom_nodes`
- download `4x-UltraSharp`, `RealESRGAN_x4plus`, and the SDXL tile controlnet
- optionally download the 6.6 GB FLUX union controlnet

The installer intentionally does not auto-download the base FLUX checkpoint. That remains manual because its license and distribution terms are separate.

## Workflow

Import `workflows/hyper_tile_sdxl.json` into ComfyUI.

What to set after import:

1. Select your source image in `LoadImage`.
2. Pick an SDXL checkpoint in `CheckpointLoaderSimple`.
3. Adjust the base prompt in `HyperTile Prompt Composer`.
4. Adjust target size or VRAM in `HyperTile Planner`.
5. Queue the workflow.

## FLUX Workflow

Import `workflows/hyper_tile_flux_tiled_refiner.json` when you want a true tiled FLUX pass instead of only using the profile file.

What it does:

1. Splits the input image into a batched tile set with `TTP_Image_Tile_Batch`.
2. Captions the tile batch with Florence-2 and merges those cues into the positive prompt.
3. Loads the fp8 FLUX dev checkpoint with `CheckpointLoaderSimple`.
4. Applies `flux_controlnet_union_pro.safetensors` through `SetUnionControlNetType` with the type fixed to `tile`.
5. Samples all tiles as a batch.
6. Reassembles the refined tiles with `TTP_Image_Assy`.

Important limits:

- This workflow expects the FLUX fp8 checkpoint file in `models/checkpoints/flux1-dev-fp8.safetensors`.
- It also expects the optional FLUX union controlnet downloaded with `--with-flux-assets`.
- The tiled FLUX workflow currently uses one merged positive prompt across the tile batch. The `HyperTile Regional Conditioning` node is included for more advanced spatial prompting, but it is not wired into the default FLUX graph because batch-wise regional prompt control is still model- and graph-dependent.

## FLUX Final Pass

The repo includes `workflows/hyper_tile_flux_refiner_profile.json` as a settings profile for the optional third pass.

Recommended use:

1. Finish the SDXL workflow first.
2. Feed the SDXL result into a FLUX tiled workflow of your choice.
3. Keep denoise low, usually `0.15` to `0.22`.
4. Use the FLUX union controlnet in tile mode at `0.3` to `0.6` strength.
5. Keep tiles at `512` on 12-16 GB cards.

If you want the actual ComfyUI graph for that stage instead of only the profile, import `workflows/hyper_tile_flux_tiled_refiner.json`.

## Limits And Notes

- `HyperTile Caption Tiles` downloads Florence-2 into the active Python cache the first time you run it.
- The bundled workflow targets PNG-style image output. EXR and WebP export require additional save nodes or extensions in your ComfyUI install.
- The bundled FLUX asset is optional because it is large and many users do not need a third refinement pass.

## License

This project is released under the MIT License. See `LICENSE`.