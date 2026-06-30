#!/usr/bin/env bash
# Fetch the ONNX Runtime + DirectML native libraries bundled into the GPU
# editions, into vendor/onnxruntime/. These libraries are large and are NOT
# committed to git (see .gitignore); run this once before building the GPU
# packages (scripts/package_*_gpu.sh).
#
# Versions are pinned to match `ort 2.0.0-rc.10` (ONNX Runtime 1.22.x).
#   - macOS arm64 : libonnxruntime.dylib (CoreML EP is built in)
#   - Windows x64 : onnxruntime.dll (DirectML EP) + onnxruntime_providers_shared.dll
#                   + DirectML.dll  (from the Microsoft.AI.DirectML package)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORT_VER="1.22.0"
DML_VER="1.15.4"

MAC="$ROOT/vendor/onnxruntime/macos-arm64"
WIN="$ROOT/vendor/onnxruntime/windows-x64"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$MAC" "$WIN"

echo "==> macOS arm64: onnxruntime $ORT_VER"
curl -fSL --retry 3 -o "$WORK/ort-mac.tgz" \
  "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-osx-arm64-${ORT_VER}.tgz"
tar xzf "$WORK/ort-mac.tgz" -C "$WORK"
SRC="$WORK/onnxruntime-osx-arm64-${ORT_VER}"
# Ship the real (versioned) dylib under the canonical name ort loads at run time.
cp "$SRC/lib/libonnxruntime.${ORT_VER}.dylib" "$MAC/libonnxruntime.dylib"
cp "$SRC/LICENSE"                "$MAC/onnxruntime-LICENSE.txt"
cp "$SRC/ThirdPartyNotices.txt"  "$MAC/onnxruntime-ThirdPartyNotices.txt"

echo "==> Windows x64: onnxruntime $ORT_VER (DirectML EP)"
curl -fSL --retry 3 -o "$WORK/ort-win.nupkg" \
  "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/${ORT_VER}"
mkdir -p "$WORK/win"; (cd "$WORK/win" && unzip -o -q ../ort-win.nupkg)
cp "$WORK/win/runtimes/win-x64/native/onnxruntime.dll"                  "$WIN/onnxruntime.dll"
cp "$WORK/win/runtimes/win-x64/native/onnxruntime_providers_shared.dll" "$WIN/onnxruntime_providers_shared.dll"
cp "$WORK/win/LICENSE"             "$WIN/onnxruntime-LICENSE.txt"
cp "$WORK/win/ThirdPartyNotices.txt" "$WIN/onnxruntime-ThirdPartyNotices.txt"

echo "==> Windows x64: DirectML $DML_VER"
curl -fSL --retry 3 -o "$WORK/dml.nupkg" \
  "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/${DML_VER}"
mkdir -p "$WORK/dml"; (cd "$WORK/dml" && unzip -o -q ../dml.nupkg)
cp "$WORK/dml/bin/x64-win/DirectML.dll" "$WIN/DirectML.dll"
cp "$WORK/dml/LICENSE.txt"              "$WIN/DirectML-LICENSE.txt"

echo
echo "=== vendor/onnxruntime/macos-arm64 ===" && ls -la "$MAC"
echo "=== vendor/onnxruntime/windows-x64 ===" && ls -la "$WIN"
echo "done."
