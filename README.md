# ComfyUI HyperTile Upscaler

This repository is a ComfyUI custom-node package plus installer bundle for large-format generative upscaling.

Repository target: `GeekatplayStudio/ComfyUI-UltimateUpsacaler`

It ships three things:

- A single-click installer that clones the missing custom nodes and downloads the core upscaler, FLUX checkpoint, and controlnet assets into the active ComfyUI install.
- Custom helper nodes for tile planning and caption-guided prompt construction.
- Ready-to-import SDXL and FLUX workflows for the resize and tiled denoise stages, plus a separate FLUX refiner profile.

## What It Builds

The default flow is:

1. Stage 1 neural resize with `4x-UltraSharp` or `RealESRGAN_x4plus`.
2. Stage 2 tiled img2img refinement with `UltimateSDUpscaleNoUpscale`.
3. Optional Stage 3 FLUX refinement using the profile in `profiles/hyper_tile_flux_refiner_profile.json`.

The tiled stage is set up around low denoise, Gaussian-style edge blending through mask blur and padding, and tiled decode to stay stable on 12-16 GB GPUs.

## Included Nodes

- `HyperTile Planner`
  - Detects the input image size, computes target output size from long edge, magnification, or exact dimensions, and recommends tile size, denoise, and tiled decode settings.
- `HyperTile Caption Tiles`
  - Optionally uses Florence-2 through `transformers` to caption the input image or tile batch and extract texture-aware prompt cues. The bundled workflows default this to `disabled` to avoid Hugging Face warning spam unless you explicitly enable captioning.
- `HyperTile Prompt Composer`
  - Merges the user prompt, tile caption prompt, and a quality suffix into a single deduplicated prompt string.
- `HyperTile Resize Image`
  - Resizes the stage-1 upscaled image to the exact planned output size before tiled denoise.
- `HyperTile Tile Preview`
  - Builds a preview overlay that shows the planned output resolution and tile boundaries before the tiled stage runs.
- `HyperTile Regional Conditioning`
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
- download `4x-UltraSharp`, `RealESRGAN_x4plus`, and the SDXL tile controlnet
- download `flux1-dev-fp8.safetensors` into `ComfyUI/models/checkpoints`
- download `flux_controlnet_union_pro.safetensors` into `ComfyUI/models/controlnet`
- optionally copy a local FLUX fp8 checkpoint into `models/checkpoints/flux1-dev-fp8.safetensors` instead of downloading it

The default launcher scripts now perform the complete model install in one click. If you run `install.py` directly, include `--with-flux-assets` to download the FLUX assets automatically.

## Workflow

Import `workflows/hyper_tile_sdxl.json` into ComfyUI.

What to set after import:

1. Select your source image in `LoadImage`.
2. Pick an SDXL checkpoint in `CheckpointLoaderSimple`.
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
7. Loads the fp8 FLUX dev checkpoint with `CheckpointLoaderSimple`.
8. Applies `flux_controlnet_union_pro.safetensors` through `SetUnionControlNetType` with the type fixed to `tile`.
9. VAE-encodes each resized tile batch and samples from that encoded latent instead of starting from an empty latent.
10. Reassembles the refined tiles with `TTP_Image_Assy`.

Important limits:

- This workflow expects the FLUX fp8 checkpoint file in `models/checkpoints/flux1-dev-fp8.safetensors`.
- It also expects the FLUX union controlnet downloaded with `--with-flux-assets`.
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

## Limits And Notes

- `HyperTile Caption Tiles` downloads Florence-2 into the active Python cache the first time you run it, but the bundled workflows ship with captioning disabled by default.
- The repeated `404`, unauthenticated Hugging Face, and Florence processor warnings come from the optional Florence caption model probing for alternate metadata files. They are not core ComfyUI model-loading errors.
- The bundled workflow targets PNG-style image output. EXR and WebP export require additional save nodes or extensions in your ComfyUI install.
- The bundled FLUX asset is optional because it is large and many users do not need a third refinement pass.

## License

This project is released under the MIT License. See `LICENSE`.