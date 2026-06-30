@echo off
chcp 65001 >nul
setlocal
rem AI Image Upscaler - 모델을 캐시에 등록해 앱에서 자동 선택되게 합니다.

set "SRC=%~dp0realesrgan-x4plus.onnx"
set "DEST=%LOCALAPPDATA%\image-upscaler\models"

if not exist "%SRC%" (
  echo [오류] 같은 폴더에서 realesrgan-x4plus.onnx 를 찾을 수 없습니다.
  echo        이 배치 파일을 모델 파일과 같은 폴더에 두고 실행하세요.
  pause
  exit /b 1
)

if not exist "%DEST%" mkdir "%DEST%"

copy /Y "%SRC%" "%DEST%\realesrgan-x4plus.onnx" >nul
if errorlevel 1 (
  echo [오류] 모델 복사에 실패했습니다.
  pause
  exit /b 1
)

echo [완료] 모델을 등록했습니다:
echo        %DEST%\realesrgan-x4plus.onnx
echo.
echo 이제 image-upscaler.exe 를 실행하면 모델이 자동 선택됩니다.
pause
