//! Inference backend selection: CPU (tract, always available) or, when built
//! with the `gpu` feature, an ONNX Runtime backend with GPU execution providers.
//!
//! The GUI/CLI talk to [`Engine`] and never care which backend is active. When
//! `prefer_gpu` is requested and the GPU backend fails to initialise (e.g. the
//! onnxruntime library isn't present), `Engine` transparently falls back to CPU.

use std::path::Path;

use anyhow::Result;
use image::RgbImage;

use crate::upscaler::model::Upscaler;

/// A loaded model behind a concrete inference backend.
pub enum Engine {
    /// Pure-Rust CPU inference (tract), tiles run in parallel across cores.
    Cpu(Upscaler),
    /// ONNX Runtime backend (GPU execution providers, CPU fallback).
    #[cfg(feature = "gpu")]
    Gpu(crate::upscaler::model_ort::OrtUpscaler),
}

impl Engine {
    /// Load a model. With the `gpu` feature and `prefer_gpu`, try the GPU backend
    /// first and fall back to CPU on any failure; otherwise load CPU directly.
    pub fn load(model_path: &Path, tile_size: u32, scale: u32, prefer_gpu: bool) -> Result<Self> {
        #[cfg(feature = "gpu")]
        if prefer_gpu {
            // ONNX Runtime aborts via panic on some failures (e.g. an
            // incompatible onnxruntime library version), so guard with
            // `catch_unwind` to keep the CPU fallback graceful instead of
            // crashing the app.
            let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::upscaler::model_ort::OrtUpscaler::load(model_path, tile_size, scale)
            }));
            match loaded {
                Ok(Ok(u)) => return Ok(Engine::Gpu(u)),
                Ok(Err(e)) => {
                    eprintln!("GPU 백엔드를 사용할 수 없어 CPU로 진행합니다: {e:#}");
                }
                Err(_) => {
                    eprintln!(
                        "GPU 백엔드 초기화 중 오류로 CPU로 진행합니다 \
                         (onnxruntime 라이브러리 누락 또는 버전 불일치일 수 있어요)."
                    );
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        let _ = prefer_gpu;

        Ok(Engine::Cpu(Upscaler::load(model_path, tile_size, scale)?))
    }

    /// Human-readable name of the active backend, for status display.
    pub fn backend_name(&self) -> &'static str {
        match self {
            Engine::Cpu(_) => "CPU (tract · 멀티스레드)",
            #[cfg(feature = "gpu")]
            Engine::Gpu(_) => "GPU (ONNX Runtime)",
        }
    }

    /// Upscale a full RGB image with the active backend.
    pub fn upscale(&self, src: &RgbImage, progress: impl FnMut(f32) + Send) -> Result<RgbImage> {
        match self {
            Engine::Cpu(u) => u.upscale(src, progress),
            #[cfg(feature = "gpu")]
            Engine::Gpu(u) => u.upscale(src, progress),
        }
    }
}
