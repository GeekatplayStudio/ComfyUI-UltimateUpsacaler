# Geekatplay Studio HyperTile Upscaler For ComfyUI

This repository packages the Geekatplay Studio tiled upscaling stack for ComfyUI: installer, helper nodes, and ready-to-import workflows for SDXL and FLUX refinement.

Repository target: `GeekatplayStudio/ComfyUI-UltimateUpsacaler`

It ships four pieces:

- A single-click installer that clones the missing custom nodes and downloads the core model stack into the active ComfyUI install.
- Geekatplay Studio helper nodes for tile planning, Florence-2 caption extraction, prompt composition, resize staging, tile preview, and regional conditioning.
- A production SDXL upscale workflow for the initial large-format pass.
- A FLUX tiled refiner workflow plus a separate FLUX refiner profile for the optional finishing pass.

## What It Builds

The default flow is:

1. Stage 1 neural resize with `4x-UltraSharp` or `RealESRGAN_x4plus`.
2. Stage 2 tiled img2img refinement with `UltimateSDUpscaleNoUpscale`.
3. Optional Stage 3 FLUX refinement using `workflows/hyper_tile_flux_tiled_refiner.json` or the profile in `profiles/hyper_tile_flux_refiner_profile.json`.

The tiled stage is set up around low denoise, Gaussian-style edge blending through mask blur and padding, and tiled decode to stay stable on 12-16 GB GPUs.

## Included Nodes

- `Geekatplay HyperTile Planner`
  - Detects the input image size, computes target output size from long edge, magnification, or exact dimensions, and recommends tile size, denoise, and tiled decode settings.
- `Geekatplay HyperTile Caption Tiles`
  - Optionally uses Florence-2 through `transformers` to caption the input image or tile batch and extract texture-aware prompt cues. The bundled workflows default this to `disabled` to avoid Hugging Face warning spam unless you explicitly enable captioning.
- `Geekatplay HyperTile Prompt Composer`
  - Merges the user prompt, tile caption prompt, and a quality suffix into a single deduplicated prompt string.
- `Geekatplay HyperTile Resize Image`
  - Resizes the stage-1 upscaled image to the exact planned output size before tiled denoise.
- `Geekatplay HyperTile Tile Preview`
  - Builds a preview overlay that shows the planned output resolution and tile boundaries before the tiled stage runs.
- `Geekatplay HyperTile Regional Conditioning`
  - Converts TTP tile positions plus caption lines into area-weighted conditioning blocks for spatial prompt control.

## Single-Click Install

On macOS, double-click `install_hyper_tile.command`.

On Windows, double-click `install_hyper_tile.bat`.

The launcher scripts perform a full install by default, including the FLUX checkpoint and FLUX controlnet assets.

From a terminal:

```bash
./install_hyper_tile.sh --comfyui /path/to/ComfyUI
```

On Windows Command Prompt:

```bat
install_hyper_tile.bat --comfyui M:\ComfyUI
```

From a terminal, the launcher scripts also default to the full install:

```bash
./install_hyper_tile.sh --comfyui /path/to/ComfyUI --with-flux-assets
```

Windows equivalent:

```bat
install_hyper_tile.bat --comfyui M:\ComfyUI --with-flux-assets
```

If you already have the FLUX fp8 checkpoint locally and want to avoid downloading it again, copy it into the exact workflow filename during install:

```bat
install_hyper_tile.bat --comfyui M:\ComfyUI --with-flux-assets --flux-checkpoint D:\models\flux1-dev-fp8.safetensors
```

The installer will:

- install this package's Python requirements
- clone `ComfyUI-Manager`, `ComfyUI_UltimateSDUpscale`, and `Comfyui_TTP_Toolset` into `ComfyUI/custom_nodes`
- download `4x-UltraSharp`, `RealESRGAN_x4plus`, `realesr-general-x4v3`, `sdxl_base_1.0.safetensors`, and the SDXL tile controlnet
- optionally download `flux1-dev-fp8.safetensors`, `flux1-schnell-fp8.safetensors`, and `flux_controlnet_union_pro.safetensors`
- optionally copy a local FLUX fp8 checkpoint into `models/checkpoints/flux1-dev-fp8.safetensors` instead of downloading it

If a remote host blocks direct model download because of license acceptance or authentication rules, the installer now prints the exact manual fallback source instead of failing mid-run without context.

The default launcher scripts now perform the complete model install in one click. If you run `install.py` directly, include `--with-flux-assets` to download the FLUX assets automatically.

## Workflow Pack

- `workflows/hyper_tile_sdxl.json`
  - Geekatplay Studio SDXL production workflow for Stage 1 resize plus Stage 2 tiled denoise.
- `workflows/hyper_tile_flux_tiled_refiner.json`
  - Geekatplay Studio FLUX tiled refinement workflow for full tile batch sampling and assembly.
- `profiles/hyper_tile_flux_refiner_profile.json`
  - FLUX settings profile for users who want to wire the final pass into their own ComfyUI graph.

## Workflow

Import `workflows/hyper_tile_sdxl.json` into ComfyUI.

What to set after import:

1. Select your source image in `LoadImage`.
2. Confirm the SDXL checkpoint you want is selected in `CheckpointLoaderSimple`.
3. Adjust the base prompt in `HyperTile Prompt Composer`.
4. In `HyperTile Planner`, choose `sizing_mode`:
  - `target_long_edge` to drive output from the longest side
  - `magnification` to upscale by a factor like `2.0x` or `4.0x`
  - `output_size` to set exact width and height
  The workflows now default to `magnification`, so the visible scale-up control is the `magnification` field on `HyperTile Planner`.
5. Review the tile overlay from `HyperTile Tile Preview` before queueing if you want to confirm the tile layout.
6. Adjust VRAM, tile behavior, or `denoise_adjust` in `HyperTile Planner` if needed. Use a small negative adjustment when you see seams and a small positive adjustment only when the result is too conservative.
7. Queue the workflow.

The workflow now detects the input image dimensions automatically and resizes the stage-1 upscaled image to the planned output resolution before `UltimateSDUpscaleNoUpscale` runs.

## FLUX Workflow

Import `workflows/hyper_tile_flux_tiled_refiner.json` when you want a true tiled FLUX pass instead of only using the profile file.

This workflow requires a recent ComfyUI build with FLUX/SD3-era core nodes such as `SetUnionControlNetType`, `ControlNetApplySD3`, and the standard VAE encode/decode nodes. If the graph does not load and ComfyUI reports missing node types for those names, update ComfyUI first.

The FLUX workflow now follows the same size-planning pattern as the SDXL workflow: it detects the input image size, preserves aspect ratio, supports long-edge, magnification, or exact output sizing, runs a neural upscale pass first, and shows a tile preview before the tiled FLUX stage.

What it does:

1. Detects the input image size and plans the supported output size while keeping the same aspect ratio.
2. Upscales the image with `4x-UltraSharp`.
3. Resizes that stage-1 result to the exact planned output size.
4. Shows a tile layout preview so you can verify the cut lines before sampling.
5. Splits the resized image into a batched tile set with `TTP_Image_Tile_Batch`.
6. Captions the tile batch with Florence-2 and merges those cues into the positive prompt.
7. Loads the fp8 FLUX checkpoint with `CheckpointLoaderSimple`.
8. Applies `flux_controlnet_union_pro.safetensors` through `SetUnionControlNetType` with the type fixed to `tile`.
9. VAE-encodes each resized tile batch and samples from that encoded latent instead of starting from an empty latent.
10. Reassembles the refined tiles with `TTP_Image_Assy`.

Important limits:

- This workflow is designed around `flux1-dev-fp8.safetensors`, but `flux1-schnell-fp8.safetensors` is also supported for faster passes.
- It expects the FLUX assets downloaded with `--with-flux-assets` unless you have already placed the exact filenames manually.
- The SDXL workflow defaults to `sd_xl_base_1.0_0.9vae.safetensors`, which matches a common ComfyUI SDXL checkpoint filename.
- The tiled FLUX workflow currently uses one merged positive prompt across the tile batch. The `HyperTile Regional Conditioning` node is included for more advanced spatial prompting, but it is not wired into the default FLUX graph because batch-wise regional prompt control is still model- and graph-dependent.

## FLUX Final Pass

The repo includes `profiles/hyper_tile_flux_refiner_profile.json` as a settings profile for the optional third pass.

This file is not a ComfyUI workflow graph and is not meant to be loaded with the workflow importer.

Recommended use:

1. Finish the SDXL workflow first.
2. Feed the SDXL result into a FLUX tiled workflow of your choice.
3. Keep denoise low, usually `0.15` to `0.22`.
4. Use the FLUX union controlnet in tile mode at `0.3` to `0.6` strength.
5. Keep tiles at `512` on 12-16 GB cards.
6. The bundled FLUX workflow imports with seam-safer defaults: visible `denoise = 0.18` on `KSampler`, `denoise_adjust = -0.03` available on `HyperTile Planner`, and `padding = 96` on `TTP_Image_Assy`.

If you want the actual ComfyUI graph for that stage instead of only the profile, import `workflows/hyper_tile_flux_tiled_refiner.json`.

## Help

- Use the SDXL workflow for the first major upscale jump.
- Use the FLUX workflow only after the image is already resized and composition-locked.
- Keep tile size at `512` on 12-16 GB GPUs unless you have validated a larger tile budget locally.
- Florence-2 will populate its cache on first use, so the first caption run is slower than later runs.

## Limits And Notes

- `HyperTile Caption Tiles` downloads Florence-2 into the active Python cache the first time you run it, but the bundled workflows ship with captioning disabled by default.
- The repeated `404`, unauthenticated Hugging Face, and Florence processor warnings come from the optional Florence caption model probing for alternate metadata files. They are not core ComfyUI model-loading errors.
- The bundled workflow targets PNG-style image output. EXR and WebP export require additional save nodes or extensions in your ComfyUI install.
- The bundled FLUX assets are optional because they are large and many users only need the SDXL production path.

## License

This project is released under the MIT License. See `LICENSE`.