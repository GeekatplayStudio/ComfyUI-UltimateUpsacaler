from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
import torch


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    array = image_tensor.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def _best_device() -> str:
    try:
        import comfy.model_management as model_management

        return str(model_management.get_torch_device())
    except Exception:
        pass

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalize_segment(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = cleaned.strip(",. ")
    return cleaned


def _dedupe_segments(segments: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for segment in segments:
        normalized = _normalize_segment(segment).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(_normalize_segment(segment))
    return output


def _split_prompt(text: str) -> List[str]:
    if not text:
        return []
    return [part for part in (_normalize_segment(item) for item in text.split(",")) if part]


def _parse_tile_captions(captions_text: str) -> List[str]:
    parsed: List[str] = []
    for line in captions_text.splitlines():
        cleaned = _normalize_segment(line)
        if not cleaned:
            continue
        if ":" in cleaned:
            _, cleaned = cleaned.split(":", 1)
            cleaned = _normalize_segment(cleaned)
        if cleaned:
            parsed.append(cleaned)
    return parsed


def _encode_text_conditioning(clip, text: str):
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


@lru_cache(maxsize=4)
def _load_caption_stack(model_name: str):
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for HyperTile captioning. Run install.py or pip install -r requirements.txt."
        ) from exc

    dtype = torch.float16 if _best_device().startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    device = _best_device()
    model.to(device)
    return processor, model, device


def _florence_task(detail_level: str) -> str:
    mapping = {
        "brief": "<CAPTION>",
        "detailed": "<DETAILED_CAPTION>",
        "material-aware": "<MORE_DETAILED_CAPTION>",
    }
    return mapping[detail_level]


def _caption_florence(image: Image.Image, model_name: str, detail_level: str, max_new_tokens: int) -> str:
    processor, model, device = _load_caption_stack(model_name)
    task_prompt = _florence_task(detail_level)
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    prepared_inputs = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            prepared_inputs[key] = value.to(device)
        else:
            prepared_inputs[key] = value

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=prepared_inputs["input_ids"],
            pixel_values=prepared_inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=image.size,
    )
    caption = parsed.get(task_prompt, generated_text)
    if isinstance(caption, list):
        caption = ", ".join(str(item) for item in caption)
    return _normalize_segment(str(caption))


class HyperTilePlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_long_edge": (
                    "INT",
                    {"default": 4096, "min": 1024, "max": 16000, "step": 64},
                ),
                "model_family": (("SDXL", "FLUX.1-dev", "FLUX.1-schnell"),),
                "vram_gb": ("INT", {"default": 12, "min": 6, "max": 80, "step": 1}),
                "preserve_composition": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = (
        "tile_width",
        "tile_height",
        "batch_size",
        "denoise",
        "tiled_decode",
        "profile",
    )
    FUNCTION = "plan"
    CATEGORY = "HyperTile/Upscale"

    def plan(
        self,
        image: torch.Tensor,
        target_long_edge: int,
        model_family: str,
        vram_gb: int,
        preserve_composition: bool,
    ):
        _, image_height, image_width, _ = image.shape
        source_long_edge = max(image_width, image_height)
        upscale_ratio = max(target_long_edge / max(source_long_edge, 1), 1.0)

        if model_family == "SDXL":
            if vram_gb >= 24:
                tile_size = 1024
                batch_size = 2
            elif vram_gb >= 16:
                tile_size = 768
                batch_size = 1
            else:
                tile_size = 512
                batch_size = 1
            denoise = 0.24 if preserve_composition else 0.32
        elif model_family == "FLUX.1-dev":
            tile_size = 768 if vram_gb >= 24 else 512
            batch_size = 1
            denoise = 0.18 if preserve_composition else 0.24
        else:
            tile_size = 768 if vram_gb >= 16 else 512
            batch_size = 1
            denoise = 0.14 if preserve_composition else 0.2

        if target_long_edge >= 8192 and vram_gb <= 16:
            tile_size = min(tile_size, 512)
            batch_size = 1

        if upscale_ratio >= 6.0 and tile_size > 512:
            tile_size = 512

        profile = {
            "image_size": {"width": int(image_width), "height": int(image_height)},
            "target_long_edge": int(target_long_edge),
            "upscale_ratio": round(upscale_ratio, 3),
            "model_family": model_family,
            "vram_gb": int(vram_gb),
            "tile_width": int(tile_size),
            "tile_height": int(tile_size),
            "batch_size": int(batch_size),
            "denoise": round(float(denoise), 3),
            "tiled_decode": True,
            "notes": "Use 12-24px mask blur and 48-96px padding for seam management.",
        }
        return (
            int(tile_size),
            int(tile_size),
            int(batch_size),
            float(round(denoise, 3)),
            True,
            json.dumps(profile, indent=2),
        )


class HyperTileCaptionTiles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (("microsoft/Florence-2-base-ft", "microsoft/Florence-2-large-ft"),),
                "detail_level": (("brief", "detailed", "material-aware"),),
                "max_tiles": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
                "max_new_tokens": ("INT", {"default": 96, "min": 16, "max": 256, "step": 8}),
                "prefix": ("STRING", {"default": "", "multiline": True}),
                "suffix": (
                    "STRING",
                    {
                        "default": "fabric weave, skin pores, clean edges, coherent details",
                        "multiline": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "captions")
    FUNCTION = "caption_tiles"
    CATEGORY = "HyperTile/Captioning"

    def caption_tiles(
        self,
        image: torch.Tensor,
        model_name: str,
        detail_level: str,
        max_tiles: int,
        max_new_tokens: int,
        prefix: str,
        suffix: str,
    ):
        captions: List[str] = []
        for index, tile in enumerate(image[:max_tiles]):
            caption = _caption_florence(
                _tensor_to_pil(tile.unsqueeze(0)).convert("RGB"),
                model_name=model_name,
                detail_level=detail_level,
                max_new_tokens=max_new_tokens,
            )
            if caption:
                captions.append(f"tile {index + 1}: {caption}")

        merged_segments = _dedupe_segments(
            [prefix] + [caption.split(": ", 1)[-1] for caption in captions] + [suffix]
        )
        prompt = ", ".join(merged_segments)
        return prompt, "\n".join(captions)


class HyperTilePromptComposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": ("STRING", {"default": "", "multiline": True}),
                "tile_prompt": ("STRING", {"default": "", "multiline": True}),
                "quality_prompt": (
                    "STRING",
                    {
                        "default": "ultra detailed, high frequency texture, realistic micro details, seamless tile transitions, no ghosting, no duplicate features",
                        "multiline": True,
                    },
                ),
                "merge_mode": (("append", "prepend", "tile_only"),),
                "dedupe": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "compose"
    CATEGORY = "HyperTile/Captioning"

    def compose(
        self,
        base_prompt: str,
        tile_prompt: str,
        quality_prompt: str,
        merge_mode: str,
        dedupe: bool,
    ):
        segments: List[str]
        if merge_mode == "tile_only":
            segments = [tile_prompt, quality_prompt]
        elif merge_mode == "prepend":
            segments = [tile_prompt, base_prompt, quality_prompt]
        else:
            segments = [base_prompt, tile_prompt, quality_prompt]

        if dedupe:
            final_segments = _dedupe_segments(
                segment
                for chunk in segments
                for segment in _split_prompt(chunk)
            )
            return (", ".join(final_segments),)

        return (", ".join(segment for segment in segments if _normalize_segment(segment)),)


class HyperTileRegionalConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "positions": ("LIST", {"forceInput": True}),
                "captions": ("STRING", {"default": "", "multiline": True}),
                "base_prompt": ("STRING", {"default": "", "multiline": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "include_base_prompt": ("BOOLEAN", {"default": True}),
                "fallback_to_base_prompt": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "base_conditioning": ("CONDITIONING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "debug_prompts")
    FUNCTION = "build_conditioning"
    CATEGORY = "HyperTile/Conditioning"

    def build_conditioning(
        self,
        clip,
        positions: List[Tuple[int, int, int, int]],
        captions: str,
        base_prompt: str,
        strength: float,
        include_base_prompt: bool,
        fallback_to_base_prompt: bool,
        base_conditioning=None,
    ):
        import node_helpers

        prompts_by_tile = _parse_tile_captions(captions)
        output_conditioning = list(base_conditioning) if base_conditioning is not None else []
        debug_lines: List[str] = []

        if not positions:
            return (output_conditioning, "")

        for index, position in enumerate(positions):
            if len(position) != 4:
                raise ValueError(f"Tile position at index {index} must have 4 values, got {position}")

            left, top, right, bottom = [int(value) for value in position]
            tile_width = max(right - left, 8)
            tile_height = max(bottom - top, 8)

            tile_caption = prompts_by_tile[index] if index < len(prompts_by_tile) else ""
            prompt_segments: List[str] = []
            if include_base_prompt and base_prompt:
                prompt_segments.extend(_split_prompt(base_prompt))
            if tile_caption:
                prompt_segments.extend(_split_prompt(tile_caption))
            elif fallback_to_base_prompt and not include_base_prompt and base_prompt:
                prompt_segments.extend(_split_prompt(base_prompt))

            prompt_text = ", ".join(_dedupe_segments(prompt_segments))
            if not prompt_text:
                continue

            encoded = _encode_text_conditioning(clip, prompt_text)
            conditioned_tile = node_helpers.conditioning_set_values(
                encoded,
                {
                    "area": (tile_height // 8, tile_width // 8, top // 8, left // 8),
                    "strength": strength,
                    "set_area_to_bounds": False,
                },
            )
            output_conditioning.extend(conditioned_tile)
            debug_lines.append(
                f"tile {index + 1}: ({left}, {top}, {right}, {bottom}) -> {prompt_text}"
            )

        return (output_conditioning, "\n".join(debug_lines))


NODE_CLASS_MAPPINGS: Dict[str, object] = {
    "HyperTilePlanner": HyperTilePlanner,
    "HyperTileCaptionTiles": HyperTileCaptionTiles,
    "HyperTilePromptComposer": HyperTilePromptComposer,
    "HyperTileRegionalConditioning": HyperTileRegionalConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HyperTilePlanner": "HyperTile Planner",
    "HyperTileCaptionTiles": "HyperTile Caption Tiles",
    "HyperTilePromptComposer": "HyperTile Prompt Composer",
    "HyperTileRegionalConditioning": "HyperTile Regional Conditioning",
}