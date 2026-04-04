# Release Notes

## Release Hardening Update

This cleanup turns the repository into a leaner release artifact instead of a working-directory snapshot.

### Code And Installer

- Fixed caption tile planning so non-square `max_tiles` values use a fuller spatial grid instead of collapsing to `isqrt(max_tiles) ** 2` tiles.
- Improved ComfyUI Python runtime discovery for Windows, Linux, and macOS layouts.
- Made pip-based installer calls explicitly non-interactive and clearer in command logs.
- Clarified that `--with-flux-assets` downloads both the FLUX checkpoint and the FLUX Union ControlNet asset.

### Packaging And Assets

- Moved Florence captioning dependencies into an optional `captioning` extra in `pyproject.toml` for pip-based installs.
- Removed the unused `realesr-general-x4v3.pth` download from the dependency manifest.
- Expanded ignore rules for editor state, caches, and build artifacts.

### Operational Notes

- `install.py` still installs the full runtime dependency set from `requirements.txt`.
- `pip install .` installs only the core runtime metadata; use `pip install .[captioning]` if you want Florence captioning in a pip-managed environment.
- Third-party custom nodes, checkpoints, and models are not covered by this repository's MIT license.