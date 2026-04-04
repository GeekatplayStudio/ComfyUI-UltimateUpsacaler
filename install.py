from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.request


REPO_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = REPO_ROOT / "manifests" / "dependencies.json"
INSTALLER_NAME = "Geekatplay Studio HyperTile Installer"
PIP_INSTALL_ARGS = ["--disable-pip-version-check", "--no-input"]


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


def discover_comfyui_python(comfy_root: Path) -> list[str]:
    candidates = [
        comfy_root / ".venv" / "Scripts" / "python.exe",
        comfy_root / ".venv" / "bin" / "python",
        comfy_root / "venv" / "Scripts" / "python.exe",
        comfy_root / "venv" / "bin" / "python",
        comfy_root.parent / "python_embeded" / "python.exe",
        comfy_root.parent / "python_embeded" / "python",
        comfy_root.parent / "python_embedded" / "python.exe",
        comfy_root.parent / "python_embedded" / "python",
    ]

    for candidate in candidates:
        if candidate.exists():
            return [str(candidate)]

    return [sys.executable]


def run_command(command: list[str], cwd: Path | None = None) -> None:
    formatted_command = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
    print(f"[run] {formatted_command}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def install_python_packages(
    python_command: list[str], package_names: list[str], dry_run: bool, cwd: Path | None = None
) -> None:
    if not package_names:
        return
    if dry_run:
        print(f"[skip] dry run: would install packages: {', '.join(package_names)}")
        return
    run_command([*python_command, "-m", "pip", "install", *PIP_INSTALL_ARGS, *package_names], cwd=cwd)


def load_pyproject_dependencies(pyproject_path: Path) -> list[str]:
    if not pyproject_path.exists():
        return []

    try:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []

    dependencies = data.get("project", {}).get("dependencies", [])
    return [dependency for dependency in dependencies if isinstance(dependency, str)]


def install_python_requirements(python_command: list[str], dry_run: bool) -> None:
    requirements_path = REPO_ROOT / "requirements.txt"
    if requirements_path.exists():
        if dry_run:
            print(f"[skip] dry run: would install {requirements_path}")
            return
        run_command([*python_command, "-m", "pip", "install", *PIP_INSTALL_ARGS, "-r", str(requirements_path)])


def run_nested_installer(node_path: Path, python_command: list[str], dry_run: bool) -> None:
    install_script = node_path / "install.py"
    requirements_path = node_path / "requirements.txt"
    pyproject_path = node_path / "pyproject.toml"
    pyproject_dependencies = load_pyproject_dependencies(pyproject_path)

    if dry_run:
        if install_script.exists() or requirements_path.exists() or pyproject_dependencies:
            print(f"[skip] dry run: would install dependencies for {node_path.name}")
        return

    if install_script.exists():
        run_command([*python_command, str(install_script)], cwd=node_path)
    elif requirements_path.exists():
        run_command([*python_command, "-m", "pip", "install", *PIP_INSTALL_ARGS, "-r", str(requirements_path)], cwd=node_path)
    elif pyproject_dependencies:
        install_python_packages(python_command, pyproject_dependencies, dry_run=dry_run, cwd=node_path)


def ensure_custom_nodes(comfy_root: Path, python_command: list[str], manifest: dict, dry_run: bool) -> None:
    custom_nodes_dir = comfy_root / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)

    for node in manifest["custom_nodes"]:
        target_dir = custom_nodes_dir / node["directory"]
        if target_dir.exists():
            print(f"[reuse] custom node exists: {target_dir}")
        else:
            print(f"[clone] {node['name']} -> {target_dir}")
            if dry_run:
                continue

            run_command(["git", "clone", "--depth", "1", node["repo"], str(target_dir)], cwd=custom_nodes_dir)

        run_nested_installer(target_dir, python_command=python_command, dry_run=dry_run)
        install_python_packages(
            python_command,
            node.get("python_packages", []),
            dry_run=dry_run,
            cwd=target_dir,
        )


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(destination.parent), suffix=".part") as temp_file:
        temp_path = Path(temp_file.name)

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": f"{INSTALLER_NAME}/0.1"},
        )
        with urllib.request.urlopen(request) as response, temp_path.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def copy_file(source: Path, destination: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[skip] dry run: would copy {source} -> {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def ensure_flux_checkpoint(
    comfy_root: Path,
    flux_checkpoint_path: str | None,
    dry_run: bool,
) -> None:
    target_path = comfy_root / "models" / "checkpoints" / "flux1-dev-fp8.safetensors"
    if target_path.exists():
        print(f"[skip] model exists: {target_path}")
        return

    if not flux_checkpoint_path:
        return

    source_path = Path(flux_checkpoint_path).expanduser().resolve()
    if dry_run:
        print(f"[copy] FLUX checkpoint -> {target_path}")
        copy_file(source_path, target_path, dry_run=dry_run)
        return

    if not source_path.exists():
        raise FileNotFoundError(f"FLUX checkpoint source does not exist: {source_path}")

    print(f"[copy] FLUX checkpoint -> {target_path}")
    copy_file(source_path, target_path, dry_run=dry_run)


def ensure_models(comfy_root: Path, manifest: dict, with_flux_assets: bool, dry_run: bool) -> list[tuple[dict, str]]:
    failures: list[tuple[dict, str]] = []
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

        try:
            download_file(model["url"], target_path)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[warn] download failed for {model['name']}: {reason}")
            failures.append((model, reason))

    return failures


def print_manual_assets(manifest: dict, download_failures: list[tuple[dict, str]]) -> None:
    if not manifest.get("manual_assets") and not download_failures:
        return

    print("\nManual assets and follow-up:")
    for asset in manifest.get("manual_assets", []):
        print(f"- {asset['name']}")
        print(f"  why: {asset['why']}")
        print(f"  source: {asset['source']}")

    for model, reason in download_failures:
        print(f"- {model['name']}")
        print(f"  why: automatic download failed ({reason})")
        print(f"  source: {model.get('source', model['url'])}")


def ensure_required_downloads(download_failures: list[tuple[dict, str]]) -> None:
    required_failures = [model for model, _reason in download_failures if model.get("required")]
    if required_failures:
        names = ", ".join(model["name"] for model in required_failures)
        raise SystemExit(f"Required assets still need manual installation: {names}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install Geekatplay Studio HyperTile dependencies into ComfyUI.")
    parser.add_argument("--comfyui", help="Path to the ComfyUI root directory.")
    parser.add_argument("--skip-nodes", action="store_true", help="Skip custom node cloning.")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads.")
    parser.add_argument(
        "--with-flux-assets",
        action="store_true",
        help="Download the optional FLUX checkpoints and FLUX Union ControlNet assets.",
    )
    parser.add_argument(
        "--flux-checkpoint",
        help="Path to a local FLUX fp8 checkpoint to copy into ComfyUI/models/checkpoints/flux1-dev-fp8.safetensors.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing anything.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest()
    comfy_root = discover_comfyui_root(args.comfyui)
    python_command = discover_comfyui_python(comfy_root)

    print(INSTALLER_NAME)
    print(f"ComfyUI root: {comfy_root}")
    print(f"Python runtime: {' '.join(python_command)}")
    install_python_requirements(python_command, dry_run=args.dry_run)

    if not args.skip_nodes:
        ensure_custom_nodes(comfy_root, python_command, manifest, dry_run=args.dry_run)

    download_failures: list[tuple[dict, str]] = []
    if not args.skip_models:
        ensure_flux_checkpoint(
            comfy_root,
            flux_checkpoint_path=args.flux_checkpoint,
            dry_run=args.dry_run,
        )
        download_failures = ensure_models(
            comfy_root,
            manifest,
            with_flux_assets=args.with_flux_assets,
            dry_run=args.dry_run,
        )

    print_manual_assets(manifest, download_failures)
    if not args.dry_run:
        ensure_required_downloads(download_failures)

    print("\nDone. Restart ComfyUI and import a workflow from workflows/.")


if __name__ == "__main__":
    main()