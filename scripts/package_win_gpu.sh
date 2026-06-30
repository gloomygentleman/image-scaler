#!/usr/bin/env bash
# Assemble the Windows (x86_64) GPU edition zip.
#   build first :  cargo build --release --target x86_64-pc-windows-gnu --features gpu
#   runtime libs:  scripts/fetch_onnxruntime.sh   (populates vendor/)
# The GPU exe path can be overridden with EXE=... (defaults to the cross-build output).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXE="${EXE:-$ROOT/target/x86_64-pc-windows-gnu/release/image-upscaler.exe}"
MODEL="$HOME/.cache/image-upscaler/models/realesrgan-x4plus.onnx"
ORT="$ROOT/vendor/onnxruntime/windows-x64"
VER=$(grep -m1 '^version' "$ROOT/Cargo.toml" | sed -E 's/.*"([^"]+)".*/\1/')
PKG="AI-Image-Upscaler-v${VER}-windows-x64-gpu"
STAGE="$ROOT/dist/$PKG"

[ -f "$EXE" ] || { echo "ERROR: exe not found at $EXE"; exit 1; }
for f in onnxruntime.dll onnxruntime_providers_shared.dll DirectML.dll; do
  [ -f "$ORT/$f" ] || { echo "ERROR: $ORT/$f not found (run scripts/fetch_onnxruntime.sh)"; exit 1; }
done

rm -rf "$STAGE"
mkdir -p "$STAGE"

cp "$EXE" "$STAGE/image-upscaler.exe"

# GPU runtime DLLs — must sit next to the exe (ort/onnxruntime load them from the app dir).
cp "$ORT/onnxruntime.dll"                  "$STAGE/onnxruntime.dll"
cp "$ORT/onnxruntime_providers_shared.dll" "$STAGE/onnxruntime_providers_shared.dll"
cp "$ORT/DirectML.dll"                     "$STAGE/DirectML.dll"

# Bundle the model so it works offline out of the box.
if [ -f "$MODEL" ]; then
  cp "$MODEL" "$STAGE/realesrgan-x4plus.onnx"
else
  echo "WARN: model not cached; distribution will require in-app download"
fi

# Text/scripts -> CRLF for Windows (DLLs/model are copied as-is above).
to_crlf() { sed 's/$/\r/' "$1" > "$2"; }
to_crlf "$ROOT/dist/사용법-windows-gpu.txt"      "$STAGE/사용법.txt"
to_crlf "$ROOT/dist/install-model.bat"           "$STAGE/install-model.bat"
to_crlf "$ROOT/dist/THIRD-PARTY-NOTICES-gpu.txt" "$STAGE/THIRD-PARTY-NOTICES.txt"
to_crlf "$ROOT/assets/fonts/OFL.txt"             "$STAGE/OFL.txt"
to_crlf "$ORT/onnxruntime-LICENSE.txt"           "$STAGE/onnxruntime-LICENSE.txt"
to_crlf "$ORT/onnxruntime-ThirdPartyNotices.txt" "$STAGE/onnxruntime-ThirdPartyNotices.txt"
to_crlf "$ORT/DirectML-LICENSE.txt"              "$STAGE/DirectML-LICENSE.txt"

# Zip it (store paths relative to dist/).
cd "$ROOT/dist"
rm -f "${PKG}.zip"
zip -r -q "${PKG}.zip" "$PKG"

echo "=== staged files ==="
ls -la "$STAGE"
echo "=== zip ==="
ls -la "$ROOT/dist/${PKG}.zip"
echo "ZIP_PATH=$ROOT/dist/${PKG}.zip"
