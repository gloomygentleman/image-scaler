#!/usr/bin/env bash
# Assemble the macOS (Apple Silicon / arm64) GPU edition zip.
#   build first :  cargo build --release --features gpu
#   runtime libs:  scripts/fetch_onnxruntime.sh   (populates vendor/)
# The GPU binary path can be overridden with BIN=... (defaults to the last
# `cargo build --release` output).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/target/release/image-upscaler}"
MODEL="$HOME/.cache/image-upscaler/models/realesrgan-x4plus.onnx"
ORT="$ROOT/vendor/onnxruntime/macos-arm64"
VER=$(grep -m1 '^version' "$ROOT/Cargo.toml" | sed -E 's/.*"([^"]+)".*/\1/')
PKG="AI-Image-Upscaler-v${VER}-macos-arm64-gpu"
STAGE="$ROOT/dist/$PKG"

[ -f "$BIN" ] || { echo "ERROR: binary not found at $BIN (run: cargo build --release --features gpu)"; exit 1; }
[ -f "$ORT/libonnxruntime.dylib" ] || { echo "ERROR: $ORT/libonnxruntime.dylib not found (run scripts/fetch_onnxruntime.sh)"; exit 1; }

rm -rf "$STAGE"
mkdir -p "$STAGE"

cp "$BIN" "$STAGE/image-upscaler"
chmod +x "$STAGE/image-upscaler"

# GPU runtime — must sit next to the binary (ort loads it relative to the exe).
cp "$ORT/libonnxruntime.dylib" "$STAGE/libonnxruntime.dylib"

# Bundle the model so it works offline out of the box.
if [ -f "$MODEL" ]; then
  cp "$MODEL" "$STAGE/realesrgan-x4plus.onnx"
else
  echo "WARN: model not cached; distribution will require in-app download"
fi

# Text/scripts -> keep LF for macOS.
cp "$ROOT/dist/사용법-macos-gpu.txt"          "$STAGE/사용법.txt"
cp "$ROOT/dist/install-model-gpu.command"      "$STAGE/install-model.command"
chmod +x "$STAGE/install-model.command"
cp "$ROOT/dist/THIRD-PARTY-NOTICES-gpu.txt"    "$STAGE/THIRD-PARTY-NOTICES.txt"
cp "$ROOT/assets/fonts/OFL.txt"                "$STAGE/OFL.txt"
cp "$ORT/onnxruntime-LICENSE.txt"              "$STAGE/onnxruntime-LICENSE.txt"
cp "$ORT/onnxruntime-ThirdPartyNotices.txt"    "$STAGE/onnxruntime-ThirdPartyNotices.txt"

# Zip it (store paths relative to dist/). -X drops extra Mac attrs for a clean archive.
cd "$ROOT/dist"
rm -f "${PKG}.zip"
zip -r -q -X "${PKG}.zip" "$PKG"

echo "=== staged files ==="
ls -la "$STAGE"
echo "=== zip ==="
ls -la "$ROOT/dist/${PKG}.zip"
echo "ZIP_PATH=$ROOT/dist/${PKG}.zip"
