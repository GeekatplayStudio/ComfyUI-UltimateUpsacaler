#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

chmod +x "$SCRIPT_DIR/install_hyper_tile.sh"
exec "$SCRIPT_DIR/install_hyper_tile.sh" "$@"