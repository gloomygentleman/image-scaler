# AI Image Upscaler (Rust)

ONNX 초해상도(Super-Resolution) 모델로 이미지를 손상 없이 업스케일하고,
이미지를 **SVG 벡터로 변환(트레이싱)** 하는 데스크톱 GUI 프로그램입니다. 기본 빌드는
**순수 Rust** 엔진만 사용하므로, 실행 파일이 **Python/런타임 설치 없이** 단독으로
동작합니다. 선택적으로 **GPU 가속 (ONNX Runtime)** 백엔드를 켤 수 있습니다
(아래 "GPU 가속" 참고).

창 오른쪽 위에서 두 모드를 전환합니다: **[업스케일]** / **[벡터 변환]**.

## 다운로드

미리 빌드된 실행 파일은 [Releases](https://github.com/gloomygentleman/image-scaler/releases/latest)
에서 받을 수 있습니다 (모델·문서 포함, 압축 해제 후 바로 실행).

각 OS마다 **CPU 전용판**과 **GPU 가속판**이 있습니다. GPU 가속판은 ONNX Runtime
런타임을 함께 동봉해, 압축을 풀고 실행하면 **별도 설치 없이 바로 GPU로 동작**합니다
(GPU를 못 쓰는 환경에서는 자동으로 CPU로 전환). 처음이라면 CPU 전용판이 가장 가볍고
간단합니다.

| OS | CPU 전용판 (가벼움, 단일 실행 파일) | GPU 가속판 (런타임 동봉, 받자마자 GPU) |
|----|----|----|
| **Windows (x64)** | `AI-Image-Upscaler-vX.Y.Z-windows-x64.zip` | `AI-Image-Upscaler-vX.Y.Z-windows-x64-gpu.zip` (DirectML) |
| **macOS (Apple Silicon)** | `AI-Image-Upscaler-vX.Y.Z-macos-arm64.zip` | `AI-Image-Upscaler-vX.Y.Z-macos-arm64-gpu.zip` (CoreML) |

> 모든 빌드는 서명/공증이 적용되지 않아 첫 실행 시 OS 보안 경고가 나타날 수 있습니다.
> Windows는 "추가 정보 → 실행", macOS는 동봉한 `install-model.command` 실행(또는
> `xattr -dr com.apple.quarantine .`)으로 진행하세요. GPU 가속판은 동봉된 런타임
> (macOS `libonnxruntime.dylib` / Windows `onnxruntime.dll`·`DirectML.dll` 등)을
> **실행 파일과 같은 폴더에** 두어야 GPU가 동작합니다. 자세한 내용은 각 패키지의
> `사용법.txt` 참고.

## 특징

- **벡터 변환 (이미지 → SVG)** — 순수 Rust `vtracer`로 트레이싱
  - 컬러/흑백, 곡선·다각형·픽셀 모드, 잡티 제거·색상 정밀도 등 조절
  - 변환 결과 SVG를 `resvg`로 즉시 렌더해 **전/후 비교 슬라이더**로 확인
  - 로고·아이콘·라인아트 등 단색 그래픽에 최적 (사진은 path가 많아져 비권장)
- AI 초해상도 추론
  - 기본: `tract`(순수 Rust, CPU) — **타일을 CPU 코어별로 병렬 처리**
  - 선택: `gpu` 기능 빌드 시 ONNX Runtime로 **GPU 가속**(macOS CoreML / Windows
    DirectML), 실패 시 자동 CPU 폴백
- **목표 크기 지정 방식 2가지**
  - **배율로**: 2× · 3× · 4× 프리셋 또는 1.0~8.0× 슬라이더
  - **크기 직접 지정**: 가로·세로 픽셀을 직접 입력 (비율 유지 옵션)
  - 모델이 고유 배율로 추론한 뒤, 목표 크기에 맞춰 고품질(Lanczos)로 재샘플링
- 다양한 **입력 포맷**: PNG, JPG/JPEG, BMP, WebP, TIFF, GIF
- 사용자가 선택하는 **출력 포맷**: PNG · JPEG(품질 조절) · WebP · BMP · TIFF
  - "입력과 동일 포맷 유지" 옵션 제공
- **드래그형 전/후 비교 미리보기**: 핸들을 끌어 원본과 결과를 한 화면에서 비교
- **모델 자동 다운로드 + 로컬 캐시** (앱 내 버튼 또는 CLI)
- 대용량 이미지 **타일 분할 처리**(이음새 없음) → 메모리 초과 방지
- **알파 채널(투명도) 보존**
- 여러 이미지 **배치 처리**
- 백그라운드 스레드 처리 → **진행률 표시 / 취소** 가능 (UI 멈춤 없음)
- 처리 전/후 **미리보기**
- **한글 폰트 내장**: 한글 UI 폰트(나눔고딕)를 실행 파일에 포함 → 어떤 PC에서도
  글자 깨짐 없이 표시 (별도 폰트 설치 불필요)

## 빌드 / 실행

[Rust](https://rustup.rs) 설치 후:

```bash
cargo run --release      # 개발 실행 (CPU, 멀티스레드)
cargo build --release    # 배포용 단일 실행 파일 빌드 (target/release/)
cargo test               # 단위 테스트 (타일링 / 포맷 / 크기 / UI 레이아웃)

cargo run --release --features gpu   # GPU 가속 빌드로 실행 (onnxruntime 필요, 아래 참고)
```

> macOS·Windows 배포 시, OS 보안 경고를 없애려면 코드 서명/공증이 별도로
> 필요합니다(언어와 무관).

## GPU 가속 (선택, `gpu` 기능)

기본 빌드는 CPU(tract)에서 타일을 코어별로 병렬 처리합니다. 더 빠른 추론이 필요하면
`gpu` 기능으로 빌드해 **ONNX Runtime** 백엔드를 사용할 수 있습니다.

- 실행 공급자(EP): **macOS = CoreML**, **Windows = DirectML**(NVIDIA/AMD/Intel 모두),
  항상 **CPU 폴백** 포함.
- ONNX Runtime 라이브러리는 **런타임에 동적 로드**(`load-dynamic`)합니다. 즉 GPU 빌드는
  더 이상 "단일 파일"이 아니며, onnxruntime 공유 라이브러리가 필요합니다(파이썬은 여전히 불필요).

**가장 쉬운 방법: GPU 가속판 다운로드.** 위 [다운로드](#다운로드)의 GPU 가속판은
onnxruntime 런타임을 실행 파일과 함께 동봉해, 압축만 풀면 바로 GPU로 동작합니다
(`ORT_DYLIB_PATH` 설정 불필요). 아래는 **직접 소스에서 빌드**할 때의 준비 방법입니다.

직접 빌드 시 준비 방법:

1. 런타임을 받아 `vendor/` 에 채웁니다:
   ```bash
   ./scripts/fetch_onnxruntime.sh   # onnxruntime 1.22.x + DirectML 1.15.x 다운로드
   ```
   (수동으로 받으려면 ort `2.0.0-rc.10`에 맞는 **onnxruntime 1.22.x**를
   <https://github.com/microsoft/onnxruntime/releases> 등에서 받으면 됩니다.)
2. GPU 기능으로 빌드하고, 런타임 라이브러리를 실행 파일과 같은 폴더에 두거나
   경로를 지정합니다:
   ```bash
   cargo build --release --features gpu
   export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib   # Windows는 onnxruntime.dll
   ```
   (`scripts/package_mac_gpu.sh` / `scripts/package_win_gpu.sh` 는 빌드 결과와
   `vendor/` 의 런타임을 묶어 배포용 zip을 만듭니다.)
3. 앱에서 **④ 모델 → 고급 설정 → "GPU 가속 사용"** 체크, 또는 CLI에 `--gpu`.

동작/주의:

- onnxruntime을 찾지 못하거나 버전이 안 맞으면 **자동으로 CPU로 폴백**합니다(앱은 죽지 않음).
- GPU 빌드는 모든 작업과 파일 저장을 마친 뒤 **프로세스 종료 시** onnxruntime의 C++
  정리 단계에서 경고가 날 수 있어, 깔끔한 종료를 위해 종료 경로를 별도로 처리합니다
  (결과물에는 영향 없음).

## 모델 준비

이 프로그램은 추론만 수행하며, **사전학습된 ONNX 모델**이 필요합니다.

### 가장 쉬운 방법: 자동 다운로드

- **앱에서**: "모델이 없나요? 자동 다운로드" 펼치기 → 모델 선택 → 다운로드.
  진행률이 표시되고, 받은 모델은 사용자 캐시에 저장되어 다음부터 즉시 사용됩니다.
- **CLI에서**: `cargo run --release -- --download 0`
  (검증된 Real-ESRGAN x4plus, 약 64MB)

캐시 위치: `~/.cache/image-upscaler/models/` (Windows: `%LOCALAPPDATA%\image-upscaler\models\`).

### 직접 준비하는 경우

`.onnx` 파일을 앱의 **"ONNX 모델 선택"** 으로 지정하고
**배율(2x/3x/4x)을 모델의 실제 배율과 동일하게** 설정하세요. 모델 규약:

- 입력: `NCHW`, RGB, float32, 값 범위 `[0, 1]`, 동적 H/W
- 출력: `NCHW`, RGB, float32, 값 범위 `[0, 1]`, 가로·세로가 `scale`배

> 참고: Real-ESRGAN `x4plus` 등이 PyTorch(.pth)로 배포되는 경우 `torch.onnx.export`로
> ONNX 변환이 필요합니다. 동적 입력 크기로 내보내면 본 프로그램이 타일 크기로
> 형태를 고정해 사용합니다.

## 명령줄(CLI) 사용

GUI 없이 자동화/배치에 쓸 수 있습니다.

```bash
# 단일 파일 업스케일: --cli <모델> <모델배율> <입력> <출력> [타일크기] [목표가로 목표세로]
image-upscaler --cli model.onnx 4 input.png output.webp 128

# 가로·세로를 직접 지정 (모델배율로 추론 후 해당 크기로 재샘플링)
image-upscaler --cli model.onnx 4 input.png output.png 128 1920 1080

# GPU 백엔드 사용 (gpu 기능으로 빌드 + onnxruntime 준비 시). 실패하면 자동 CPU 폴백
image-upscaler --cli model.onnx 4 input.png output.png 128 --gpu

# 모델 사전 다운로드 (인덱스 생략 시 0)
image-upscaler --download 0

# 벡터 변환(SVG): --vectorize <입력> <출력.svg> [color|binary] [pixel|polygon|spline] [최대크기]
image-upscaler --vectorize logo.png logo.svg
image-upscaler --vectorize logo.png logo.svg color spline 1000
image-upscaler --vectorize sketch.png sketch.svg binary polygon 800
```

## 사용 순서

1. **① 이미지** — 이미지(여러 장 가능)와 출력 폴더 선택
2. **② 목표 크기** — "배율로" 또는 "크기 직접 지정" 중 선택 (결과 크기가 실시간 표시)
3. **③ 출력** — 포맷/품질/파일명 접미사 설정
4. **④ 모델** — ONNX 모델 선택 또는 자동 다운로드, 고급 설정에서 모델 배율·타일 크기
5. **업스케일 시작** — 진행률을 보며, 필요하면 취소. 결과는 전/후 비교로 확인

## 프로젝트 구조

```
assets/
└── fonts/
    ├── NanumGothic-Regular.ttf  # 빌드 시 바이너리에 내장되는 한글 폰트
    └── OFL.txt                  # 폰트 라이선스 (SIL Open Font License 1.1)
src/
├── main.rs             # 진입점 (eframe) + CLI(--cli / --vectorize / --download)
├── fonts.rs            # 한글 폰트 내장/등록 (include_bytes!) + 테스트
├── theme.rs            # 색상 팔레트 · 타이포 · egui 스타일
├── app.rs              # GUI: 모드 전환, 카드 레이아웃, 전/후 비교, 배경 스레드
├── vectorize.rs        # 이미지→SVG 트레이싱(vtracer) + SVG 렌더(resvg) + 테스트
└── upscaler/
    ├── mod.rs          # 출력 경로/포맷 결정 + 테스트
    ├── engine.rs       # 백엔드 선택(CPU/GPU) + 패닉-안전 폴백
    ├── image_io.rs     # 다중 포맷 로드/저장 (품질) + 테스트
    ├── sizing.rs       # 목표 크기 계산(배율/크기 지정) + 테스트
    ├── tiling.rs       # 타일 분할/병합(이음새 제거) + 코어별 병렬 + 테스트
    ├── model.rs        # ONNX 추론 (tract, CPU, 멀티스레드)
    ├── model_ort.rs    # ONNX 추론 (ONNX Runtime, GPU) — `gpu` 기능에서만
    ├── pipeline.rs     # 로드→업스케일→목표크기 재샘플→저장 + 알파 보존
    └── download.rs     # 모델 자동 다운로드/캐시 + 테스트
scripts/
├── make_test_model.py    # (개발용) 검증 모델/샘플 생성 + 결과 검증
├── fetch_onnxruntime.sh  # GPU 동봉용 onnxruntime/DirectML 런타임 다운로드 → vendor/
├── package_win.sh        # Windows CPU 배포 zip 생성 (cross-build 후)
├── package_mac.sh        # macOS(arm64) CPU 배포 zip 생성
├── package_win_gpu.sh    # Windows GPU 배포 zip 생성 (런타임 동봉)
└── package_mac_gpu.sh    # macOS(arm64) GPU 배포 zip 생성 (런타임 동봉)
```

## 폰트 라이선스

UI 한글 표시는 **나눔고딕(NanumGothic)** © NAVER Corp. 폰트를 실행 파일에 내장해
사용합니다. **SIL Open Font License 1.1** 하에 배포되며, 전문은
`assets/fonts/OFL.txt` 에 포함되어 있습니다. OFL은 애플리케이션에 폰트를 내장·재배포
하는 것을 허용합니다.

## 범위 밖

- 동영상 업스케일링
- 모델 학습/파인튜닝 (사전학습 가중치 추론만)
- 클라우드/웹 버전
