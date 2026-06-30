#!/usr/bin/env bash
# Assemble the macOS (Apple Silicon / arm64) distribution zip.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/target/release/image-upscaler}"
MODEL="$HOME/.cache/image-upscaler/models/realesrgan-x4plus.onnx"
VER=$(grep -m1 '^version' "$ROOT/Cargo.toml" | sed -E 's/.*"([^"]+)".*/\1/')
PKG="AI-Image-Upscaler-v${VER}-macos-arm64"
STAGE="$ROOT/dist/$PKG"

[ -f "$BIN" ] || { echo "ERROR: binary not found at $BIN (run: cargo build --release)"; exit 1; }

rm -rf "$STAGE"
mkdir -p "$STAGE"

cp "$BIN" "$STAGE/image-upscaler"
chmod +x "$STAGE/image-upscaler"

# Bundle the model so it works offline out of the box.
if [ -f "$MODEL" ]; then
  cp "$MODEL" "$STAGE/realesrgan-x4plus.onnx"
else
  echo "WARN: model not cached; distribution will require in-app download"
fi

# Text/scripts -> keep LF for macOS.
cp "$ROOT/dist/사용법-macos.txt"          "$STAGE/사용법.txt"
cp "$ROOT/dist/install-model.command"      "$STAGE/install-model.command"
chmod +x "$STAGE/install-model.command"
cp "$ROOT/dist/THIRD-PARTY-NOTICES.txt"    "$STAGE/THIRD-PARTY-NOTICES.txt"
cp "$ROOT/assets/fonts/OFL.txt"            "$STAGE/OFL.txt"

# Zip it (store paths relative to dist/). -X drops extra Mac attrs for a clean archive.
cd "$ROOT/dist"
rm -f "${PKG}.zip"
zip -r -q -X "${PKG}.zip" "$PKG"

echo "=== staged files ==="
ls -la "$STAGE"
echo "=== zip ==="
ls -la "$ROOT/dist/${PKG}.zip"
echo "ZIP_PATH=$ROOT/dist/${PKG}.zip"
