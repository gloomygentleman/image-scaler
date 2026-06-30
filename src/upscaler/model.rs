//! ONNX super-resolution inference using `tract` (pure Rust, CPU).
//!
//! The model is loaded with a fixed tile input shape (`1 x 3 x tile x tile`),
//! which lets `tract` fully optimize the graph and keeps memory bounded. Real
//! images are processed tile-by-tile via [`crate::upscaler::tiling`].
//!
//! Expected model I/O (matches Real-ESRGAN-style exports):
//!   input:  NCHW, RGB, float32 in `[0, 1]`
//!   output: NCHW, RGB, float32 in `[0, 1]`, spatial dims multiplied by `scale`

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use image::{Rgb, RgbImage};
use tract_onnx::prelude::*;

use std::sync::Mutex;

use crate::upscaler::tiling::{upscale_with_tiles_parallel, TilingConfig};

type RunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// A loaded super-resolution model ready to upscale images.
pub struct Upscaler {
    model: RunnableModel,
    tile_size: u32,
    scale: u32,
    overlap: u32,
}

impl Upscaler {
    /// Load an ONNX model and fix its input to `tile_size` square tiles.
    ///
    /// `scale` must match the model's native upscaling factor (e.g. 4 for an
    /// x4 model). `tile_size` trades speed (larger = fewer tiles) against memory.
    pub fn load(model_path: &Path, tile_size: u32, scale: u32) -> Result<Self> {
        let tile = tile_size.max(8) as usize;

        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .with_context(|| format!("ONNX 모델을 읽을 수 없습니다: {}", model_path.display()))?
            .with_input_fact(0, f32::fact([1, 3, tile, tile]).into())
            .context("모델 입력 형태(1x3xNxN)를 설정할 수 없습니다")?
            .into_optimized()
            .context("모델을 최적화할 수 없습니다 (지원되지 않는 연산일 수 있습니다)")?
            .into_runnable()
            .context("모델을 실행 형태로 변환할 수 없습니다")?;

        let overlap = (tile_size / 8).clamp(4, tile_size / 2);

        Ok(Self {
            model,
            tile_size,
            scale: scale.max(1),
            overlap,
        })
    }

    /// Upscale a full RGB image, tiling internally and running tiles in parallel
    /// across CPU cores. `progress` receives values in `0.0..=1.0` (it is called
    /// from worker threads as tiles finish, so values may arrive out of order).
    pub fn upscale(&self, src: &RgbImage, progress: impl FnMut(f32) + Send) -> Result<RgbImage> {
        let cfg = TilingConfig {
            tile_size: self.tile_size,
            overlap: self.overlap,
            scale: self.scale,
        };

        // `Mutex` makes the per-tile error sink and the caller's progress
        // callback usable from rayon's worker threads.
        let tile_error: Mutex<Option<anyhow::Error>> = Mutex::new(None);
        let progress = Mutex::new(progress);

        let out = upscale_with_tiles_parallel(
            src,
            &cfg,
            |tile| match self.run_tile(tile) {
                Ok(t) => t,
                Err(e) => {
                    let mut slot = tile_error.lock().unwrap();
                    if slot.is_none() {
                        *slot = Some(e);
                    }
                    // Placeholder so dimensions stay consistent; the error is
                    // surfaced after the loop and the result is discarded.
                    RgbImage::new(tile.width() * self.scale, tile.height() * self.scale)
                }
            },
            |p| {
                if let Ok(mut f) = progress.lock() {
                    f(p);
                }
            },
        );

        if let Some(e) = tile_error.into_inner().unwrap() {
            return Err(e);
        }
        Ok(out)
    }

    /// Run the model on a single full-size tile.
    fn run_tile(&self, tile: &RgbImage) -> Result<RgbImage> {
        let ts = self.tile_size as usize;
        debug_assert_eq!(tile.dimensions(), (self.tile_size, self.tile_size));

        // HWC u8 -> NCHW f32 in [0, 1].
        let input = tract_ndarray::Array4::<f32>::from_shape_fn((1, 3, ts, ts), |(_, c, y, x)| {
            tile.get_pixel(x as u32, y as u32)[c] as f32 / 255.0
        });
        let tensor: Tensor = input.into();

        let result = self
            .model
            .run(tvec!(tensor.into()))
            .context("모델 추론에 실패했습니다")?;

        let view = result
            .first()
            .ok_or_else(|| anyhow!("모델이 출력을 반환하지 않았습니다"))?
            .to_array_view::<f32>()
            .context("모델 출력을 해석할 수 없습니다")?;

        let shape = view.shape();
        if shape.len() != 4 {
            return Err(anyhow!(
                "예상치 못한 출력 형태입니다: {:?} (NCHW 4차원이어야 합니다)",
                shape
            ));
        }
        let (oh, ow) = (shape[2], shape[3]);

        let mut img = RgbImage::new(ow as u32, oh as u32);
        for y in 0..oh {
            for x in 0..ow {
                let r = to_u8(view[[0, 0, y, x]]);
                let g = to_u8(view[[0, 1, y, x]]);
                let b = to_u8(view[[0, 2, y, x]]);
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
