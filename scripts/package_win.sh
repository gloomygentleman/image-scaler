#!/usr/bin/env bash
# Assemble the Windows (x86_64) distribution zip.
# Requires a prior cross-build: cargo build --release --target x86_64-pc-windows-gnu
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXE="${EXE:-$ROOT/target/x86_64-pc-windows-gnu/release/image-upscaler.exe}"
MODEL="$HOME/.cache/image-upscaler/models/realesrgan-x4plus.onnx"
VER=$(grep -m1 '^version' "$ROOT/Cargo.toml" | sed -E 's/.*"([^"]+)".*/\1/')
PKG="AI-Image-Upscaler-v${VER}-windows-x64"
STAGE="$ROOT/dist/$PKG"

[ -f "$EXE" ] || { echo "ERROR: exe not found at $EXE"; exit 1; }

rm -rf "$STAGE"
mkdir -p "$STAGE"

cp "$EXE" "$STAGE/image-upscaler.exe"

# Bundle the model so it works offline out of the box.
if [ -f "$MODEL" ]; then
  cp "$MODEL" "$STAGE/realesrgan-x4plus.onnx"
else
  echo "WARN: model not cached; distribution will require in-app download"
fi

# Text/scripts -> CRLF for Windows.
to_crlf() { sed 's/$/\r/' "$1" > "$2"; }
to_crlf "$ROOT/dist/사용법.txt"               "$STAGE/사용법.txt"
to_crlf "$ROOT/dist/install-model.bat"        "$STAGE/install-model.bat"
to_crlf "$ROOT/dist/THIRD-PARTY-NOTICES.txt"  "$STAGE/THIRD-PARTY-NOTICES.txt"
to_crlf "$ROOT/assets/fonts/OFL.txt"          "$STAGE/OFL.txt"

# Zip it (store paths relative to dist/).
cd "$ROOT/dist"
rm -f "${PKG}.zip"
zip -r -q "${PKG}.zip" "$PKG"

echo "=== staged files ==="
ls -la "$STAGE"
echo "=== zip ==="
ls -la "$ROOT/dist/${PKG}.zip"
echo "ZIP_PATH=$ROOT/dist/${PKG}.zip"
