#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v python3 >/dev/null 2>&1; then
	PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
	PYTHON_BIN="python"
else
	echo "Python was not found in PATH."
	echo "Install Python or use the Python environment that runs ComfyUI, then try again."
	exit 1
fi

exec "$PYTHON_BIN" install.py --with-flux-assets "$@"