#!/usr/bin/env bash
# AI Image Upscaler — 모델을 캐시에 등록하고, 앱의 Gatekeeper 검역(quarantine) 속성을 제거합니다.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$DIR/realesrgan-x4plus.onnx"
DEST="${XDG_CACHE_HOME:-$HOME/.cache}/image-upscaler/models"

echo "AI Image Upscaler 설치 도우미"
echo

# 1) 모델을 캐시에 등록 (앱에서 자동 선택되게)
if [ -f "$SRC" ]; then
  mkdir -p "$DEST"
  cp -f "$SRC" "$DEST/realesrgan-x4plus.onnx"
  echo "[완료] 모델 등록: $DEST/realesrgan-x4plus.onnx"
else
  echo "[건너뜀] 같은 폴더에 realesrgan-x4plus.onnx 가 없어 모델 등록을 건너뜁니다."
  echo "         (앱에서 '자동 다운로드'로 받거나 직접 모델을 지정해도 됩니다.)"
fi

# 2) 실행 파일 검역 속성 제거 — 서명되지 않은 앱의 Gatekeeper 차단 방지
APP="$DIR/image-upscaler"
if [ -f "$APP" ]; then
  chmod +x "$APP" || true
  xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true
  echo "[완료] 실행 파일 준비: $APP"
fi

echo
echo "이제 같은 폴더의 image-upscaler 를 실행하세요"
echo "  - Finder에서 더블클릭, 또는"
echo "  - 터미널에서:  ./image-upscaler"
echo
read -n 1 -s -r -p "아무 키나 누르면 창이 닫힙니다..."
echo
