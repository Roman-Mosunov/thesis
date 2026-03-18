#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QUARTO="/mnt/c/Users/Roman/AppData/Local/Programs/Quarto/bin/quarto.exe"
INPUT_WIN="$(wslpath -w "$SCRIPT_DIR/Analysis_Presentation.qmd")"

if [[ ! -x "$QUARTO" ]]; then
  echo "Quarto not found at: $QUARTO" >&2
  exit 1
fi

"$QUARTO" render "$INPUT_WIN"
