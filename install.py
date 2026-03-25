from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request


REPO_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = REPO_ROOT / "manifests" / "dependencies.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def is_comfyui_root(path: Path) -> bool:
    return (path / "custom_nodes").is_dir() and (path / "models").is_dir()


def discover_comfyui_root(explicit_path: str | None) -> Path:
    candidates = []

    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())

    env_path = os.getenv("COMFYUI_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())

    candidates.append(Path.cwd().resolve())
    candidates.append(REPO_ROOT)
    candidates.extend(REPO_ROOT.parents)

    for candidate in candidates:
        if is_comfyui_root(candidate):
            return candidate
        if candidate.name == "custom_nodes" and is_comfyui_root(candidate.parent):
            return candidate.parent

    raise RuntimeError(
        "Could not locate your ComfyUI root. Pass --comfyui /path/to/ComfyUI or set COMFYUI_PATH."
    )


def run_command(command: list[str], cwd: Path | None = None) -> None:
    print(f"[run] {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def install_python_requirements(dry_run: bool) -> None:
    requirements_path = REPO_ROOT / "requirements.txt"
    if requirements_path.exists():
        if dry_run:
            print(f"[skip] dry run: would install {requirements_path}")
            return
        run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])


def run_nested_installer(node_path: Path, dry_run: bool) -> None:
    install_script = node_path / "install.py"
    requirements_path = node_path / "requirements.txt"

    if dry_run:
        if install_script.exists() or requirements_path.exists():
            print(f"[skip] dry run: would install dependencies for {node_path.name}")
        return

    if install_script.exists():
        run_command([sys.executable, str(install_script)], cwd=node_path)
    elif requirements_path.exists():
        run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], cwd=node_path)


def ensure_custom_nodes(comfy_root: Path, manifest: dict, dry_run: bool) -> None:
    custom_nodes_dir = comfy_root / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)

    for node in manifest["custom_nodes"]:
        target_dir = custom_nodes_dir / node["directory"]
        if target_dir.exists():
            print(f"[skip] custom node exists: {target_dir}")
            continue

        print(f"[clone] {node['name']} -> {target_dir}")
        if dry_run:
            continue

        run_command(["git", "clone", "--depth", "1", node["repo"], str(target_dir)], cwd=custom_nodes_dir)
        run_nested_installer(target_dir, dry_run=dry_run)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(destination.parent), suffix=".part") as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with urllib.request.urlopen(url) as response, temp_path.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def ensure_models(comfy_root: Path, manifest: dict, with_flux_assets: bool, dry_run: bool) -> None:
    for model in manifest["models"]:
        optional_group = model.get("optional_group")
        if optional_group == "flux" and not with_flux_assets:
            print(f"[skip] optional FLUX asset: {model['name']}")
            continue

        target_path = comfy_root / model["directory"] / model["name"]
        if target_path.exists():
            print(f"[skip] model exists: {target_path}")
            continue

        print(f"[download] {model['name']} ({model['size']}) -> {target_path}")
        if dry_run:
            continue

        download_file(model["url"], target_path)


def print_manual_assets(manifest: dict) -> None:
    print("\nManual assets:")
    for asset in manifest.get("manual_assets", []):
        print(f"- {asset['name']}")
        print(f"  why: {asset['why']}")
        print(f"  source: {asset['source']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install HyperTile Upscaler dependencies into ComfyUI.")
    parser.add_argument("--comfyui", help="Path to the ComfyUI root directory.")
    parser.add_argument("--skip-nodes", action="store_true", help="Skip custom node cloning.")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads.")
    parser.add_argument("--with-flux-assets", action="store_true", help="Download the optional 6.6 GB FLUX ControlNet Union asset.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing anything.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest()
    comfy_root = discover_comfyui_root(args.comfyui)

    print(f"ComfyUI root: {comfy_root}")
    install_python_requirements(dry_run=args.dry_run)

    if not args.skip_nodes:
        ensure_custom_nodes(comfy_root, manifest, dry_run=args.dry_run)

    if not args.skip_models:
        ensure_models(comfy_root, manifest, with_flux_assets=args.with_flux_assets, dry_run=args.dry_run)

    print_manual_assets(manifest)
    print("\nDone. Restart ComfyUI and import workflows/hyper_tile_sdxl.json.")


if __name__ == "__main__":
    main()