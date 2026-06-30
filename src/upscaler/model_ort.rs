//! Optional GPU/accelerated backend via ONNX Runtime (the `ort` crate).
//!
//! Enabled by the `gpu` Cargo feature. Execution providers are chosen per
//! platform — CoreML on macOS, DirectML on Windows — always with a CPU fallback,
//! so a session still loads if no GPU provider is available.
//!
//! ONNX Runtime is loaded dynamically (`load-dynamic`): the native library is
//! found at run time via `ORT_DYLIB_PATH` (or a system path). If it can't be
//! found or initialised, [`crate::upscaler::engine::Engine`] falls back to the
//! pure-Rust CPU backend.
//!
//! Tiles are run sequentially through a single session (the GPU is one device);
//! the model contract is identical to the CPU backend (NCHW RGB f32 in `[0,1]`).

use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use image::{Rgb, RgbImage};
use ort::session::Session;
use ort::value::Tensor;

use crate::upscaler::tiling::{upscale_with_tiles, TilingConfig};

/// A super-resolution model running on ONNX Runtime.
pub struct OrtUpscaler {
    session: Mutex<Session>,
    input_name: String,
    tile_size: u32,
    scale: u32,
    overlap: u32,
}

impl OrtUpscaler {
    /// Load `model_path` into an ONNX Runtime session with platform GPU
    /// execution providers (CPU fallback). Errors if the onnxruntime library or
    /// the model cannot be loaded — the caller can then fall back to CPU.
    pub fn load(model_path: &Path, tile_size: u32, scale: u32) -> Result<Self> {
        use ort::execution_providers::CPUExecutionProvider;
        #[cfg(target_os = "macos")]
        use ort::execution_providers::CoreMLExecutionProvider;
        #[cfg(target_os = "windows")]
        use ort::execution_providers::DirectMLExecutionProvider;

        #[allow(unused_mut)]
        let mut eps = Vec::new();
        #[cfg(target_os = "macos")]
        eps.push(CoreMLExecutionProvider::default().build());
        #[cfg(target_os = "windows")]
        eps.push(DirectMLExecutionProvider::default().build());
        eps.push(CPUExecutionProvider::default().build());

        let mut builder = Session::builder().map_err(|e| anyhow!("ORT 세션 빌더 생성 실패: {e}"))?;
        builder = builder
            .with_execution_providers(eps)
            .map_err(|e| anyhow!("실행 공급자(EP) 설정 실패: {e}"))?;
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("ONNX 모델 로드 실패: {e}"))?;

        let input_name = session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .ok_or_else(|| anyhow!("모델에 입력이 없습니다"))?;

        let overlap = (tile_size / 8).clamp(4, (tile_size / 2).max(4));

        Ok(Self {
            session: Mutex::new(session),
            input_name,
            tile_size,
            scale: scale.max(1),
            overlap,
        })
    }

    /// Upscale a full RGB image, tiling internally. `progress` receives values in
    /// `0.0..=1.0`. Tiles run sequentially through the single GPU session.
    pub fn upscale(&self, src: &RgbImage, progress: impl FnMut(f32)) -> Result<RgbImage> {
        let cfg = TilingConfig {
            tile_size: self.tile_size,
            overlap: self.overlap,
            scale: self.scale,
        };

        let mut tile_error: Option<anyhow::Error> = None;
        let out = upscale_with_tiles(
            src,
            &cfg,
            |tile| match self.run_tile(tile) {
                Ok(t) => t,
                Err(e) => {
                    if tile_error.is_none() {
                        tile_error = Some(e);
                    }
                    RgbImage::new(tile.width() * self.scale, tile.height() * self.scale)
                }
            },
            progress,
        );

        if let Some(e) = tile_error {
            return Err(e);
        }
        Ok(out)
    }

    /// Run the model on a single full-size tile.
    fn run_tile(&self, tile: &RgbImage) -> Result<RgbImage> {
        let ts = self.tile_size as usize;

        // HWC u8 -> NCHW f32 in [0, 1], flat row-major.
        let mut data = vec![0f32; 3 * ts * ts];
        for y in 0..ts {
            for x in 0..ts {
                let px = tile.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    data[c * ts * ts + y * ts + x] = px[c] as f32 / 255.0;
                }
            }
        }

        let tensor = Tensor::from_array((vec![1i64, 3, ts as i64, ts as i64], data))
            .map_err(|e| anyhow!("입력 텐서 생성 실패: {e}"))?;

        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| anyhow!("ORT 추론 실패: {e}"))?;

        let (shape, vals) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("출력 텐서 해석 실패: {e}"))?;
        if shape.len() != 4 {
            return Err(anyhow!("예상치 못한 출력 형태: {:?} (NCHW 4차원이어야 함)", &shape[..]));
        }
        let (oh, ow) = (shape[2] as usize, shape[3] as usize);

        let plane = oh * ow;
        let mut img = RgbImage::new(ow as u32, oh as u32);
        for y in 0..oh {
            for x in 0..ow {
                let idx = y * ow + x;
                let r = to_u8(vals[idx]);
                let g = to_u8(vals[plane + idx]);
                let b = to_u8(vals[2 * plane + idx]);
                img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }
        Ok(img)
    }
}

/// Clamp a model output value in `[0, 1]` to an 8-bit channel.
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}
