//! End-to-end processing: load any format -> upscale -> resize to target -> save.
//!
//! RGB channels go through the AI model at its native scale. The super-resolved
//! image is then resampled (high-quality Lanczos) to the exact target size the
//! user asked for — by factor or by explicit dimensions. An alpha channel, if
//! present, is upscaled separately and re-attached, so transparency is preserved
//! without forcing the model to handle 4 channels.

use std::path::Path;

use anyhow::Result;
use image::imageops::FilterType;
use image::{DynamicImage, GrayImage, Luma, Rgba, RgbaImage};

use crate::upscaler::engine::Engine;
use crate::upscaler::image_io::{self, OutputFormat};
use crate::upscaler::sizing::TargetSize;

/// Load `input`, upscale it to `target`, and write the result to `output`.
pub fn process(
    upscaler: &Engine,
    input: &Path,
    output: &Path,
    target: TargetSize,
    format: OutputFormat,
    quality: u8,
    progress: impl FnMut(f32) + Send,
) -> Result<()> {
    let img = image_io::load_image(input)?;
    let (tw, th) = target.resolve(img.width(), img.height());
    let result = upscale_to(upscaler, &img, tw, th, progress)?;
    image_io::save_image(&result, output, format, quality)?;
    Ok(())
}

/// Upscale an in-memory image to exactly `target_w` × `target_h`, preserving an
/// alpha channel if it has one.
///
/// The AI model runs at its native scale, then the result is resampled to the
/// requested size. When the requested size already equals the model's native
/// output, no resampling happens (the AI result is used verbatim).
pub fn upscale_to(
    upscaler: &Engine,
    img: &DynamicImage,
    target_w: u32,
    target_h: u32,
    progress: impl FnMut(f32) + Send,
) -> Result<DynamicImage> {
    let has_alpha = img.color().has_alpha();
    let rgb = img.to_rgb8();
    let sr = upscaler.upscale(&rgb, progress)?;

    // Resample the super-resolved RGB to the exact requested size.
    let rgb_out = if sr.dimensions() == (target_w, target_h) {
        sr
    } else {
        image::imageops::resize(&sr, target_w, target_h, FilterType::Lanczos3)
    };

    if !has_alpha {
        return Ok(DynamicImage::ImageRgb8(rgb_out));
    }

    // Take alpha straight from the source and resample it to the target size.
    let (w, h) = (img.width(), img.height());
    let rgba = img.to_rgba8();
    let mut alpha = GrayImage::new(w, h);
    for (x, y, p) in rgba.enumerate_pixels() {
        alpha.put_pixel(x, y, Luma([p[3]]));
    }
    let alpha_up = image::imageops::resize(&alpha, target_w, target_h, FilterType::Lanczos3);

    // Recombine resampled RGB with resampled alpha.
    let mut out = RgbaImage::new(target_w, target_h);
    for (x, y, p) in rgb_out.enumerate_pixels() {
        let a = alpha_up.get_pixel(x, y)[0];
        out.put_pixel(x, y, Rgba([p[0], p[1], p[2], a]));
    }
    Ok(DynamicImage::ImageRgba8(out))
}
