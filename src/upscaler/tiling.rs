//! Seamless tiled processing for large images.
//!
//! A super-resolution model runs on fixed-size tiles. To avoid running out of
//! memory on big images — and to avoid visible seams between tiles — we:
//!
//! 1. Cut the source into tiles of exactly `tile_size`, each one carrying an
//!    `overlap` border of context on every side (sampled with reflection at the
//!    image edges, so every tile is full-size).
//! 2. Upscale each tile independently.
//! 3. Keep only the central `step = tile_size - 2*overlap` region of every
//!    upscaled tile (scaled up), which is the part with full context, and lay
//!    those central regions side by side to reconstruct the output.
//!
//! Tiles are independent, so they can be upscaled in parallel across CPU cores
//! ([`upscale_with_tiles_parallel`]); the sequential [`upscale_with_tiles`] is
//! kept as the reference implementation the unit tests pin behaviour against.
//!
//! With an identity upscaler (`scale = 1`) this reconstructs the input exactly.

use image::{Rgb, RgbImage};

/// Geometry for tiled upscaling.
#[derive(Clone, Copy, Debug)]
pub struct TilingConfig {
    /// Side length of each (square) tile fed to the model, in input pixels.
    pub tile_size: u32,
    /// Context border kept around each tile, in input pixels.
    pub overlap: u32,
    /// Upscaling factor produced by the model.
    pub scale: u32,
}

impl TilingConfig {
    /// Pixels of new content each tile contributes along one axis (input pixels).
    fn step(&self) -> u32 {
        self.tile_size.saturating_sub(2 * self.overlap).max(1)
    }
}

/// Reflect an out-of-range coordinate back into `0..len` (mirror padding).
fn reflect(coord: i64, len: i64) -> u32 {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len - 1);
    let mut c = coord % period;
    if c < 0 {
        c += period;
    }
    if c >= len {
        c = period - c;
    }
    c as u32
}

/// Source-space top-left of every tile's *core* region, row-major.
fn tile_origins(w: u32, h: u32, cfg: &TilingConfig) -> Vec<(u32, u32)> {
    let step = cfg.step();
    let tiles_x = w.div_ceil(step).max(1);
    let tiles_y = h.div_ceil(step).max(1);
    let mut origins = Vec::with_capacity((tiles_x * tiles_y) as usize);
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            origins.push((tx * step, ty * step));
        }
    }
    origins
}

/// Build one full `tile_size` × `tile_size` input tile, sampling with reflection
/// so border tiles are still complete.
fn extract_tile(src: &RgbImage, cfg: &TilingConfig, core_x: u32, core_y: u32) -> RgbImage {
    let (w, h) = (src.width(), src.height());
    let tile = cfg.tile_size.max(1);
    let overlap = cfg.overlap.min(tile / 2);

    let mut input = RgbImage::new(tile, tile);
    for dy in 0..tile {
        let sy = reflect(core_y as i64 - overlap as i64 + dy as i64, h as i64);
        for dx in 0..tile {
            let sx = reflect(core_x as i64 - overlap as i64 + dx as i64, w as i64);
            input.put_pixel(dx, dy, *src.get_pixel(sx, sy));
        }
    }
    input
}

/// Copy the central region (skipping the scaled overlap border) of `upscaled`
/// into `out` at the matching position, clipped to the canvas.
fn blit_central(out: &mut RgbImage, upscaled: &RgbImage, cfg: &TilingConfig, core_x: u32, core_y: u32) {
    let scale = cfg.scale.max(1);
    let tile = cfg.tile_size.max(1);
    let overlap = cfg.overlap.min(tile / 2);
    let step = cfg.step();

    let ov_s = overlap * scale;
    let core_w = (step * scale).min(out.width() - core_x * scale);
    let core_h = (step * scale).min(out.height() - core_y * scale);
    for dy in 0..core_h {
        let oy = core_y * scale + dy;
        let src_y = ov_s + dy;
        for dx in 0..core_w {
            let ox = core_x * scale + dx;
            let src_x = ov_s + dx;
            let px = if src_x < upscaled.width() && src_y < upscaled.height() {
                *upscaled.get_pixel(src_x, src_y)
            } else {
                Rgb([0, 0, 0])
            };
            out.put_pixel(ox, oy, px);
        }
    }
}

/// Upscale `src` tile-by-tile (sequentially) and stitch the result back together
/// seamlessly. `upscale` is called once per tile with a full `tile_size` ×
/// `tile_size` image and must return a `(tile_size*scale)`-square image.
/// `progress` is called with a value in `0.0..=1.0` as tiles complete.
pub fn upscale_with_tiles<F, P>(src: &RgbImage, cfg: &TilingConfig, mut upscale: F, mut progress: P) -> RgbImage
where
    F: FnMut(&RgbImage) -> RgbImage,
    P: FnMut(f32),
{
    let scale = cfg.scale.max(1);
    let origins = tile_origins(src.width(), src.height(), cfg);
    let total = origins.len().max(1) as f32;

    let mut out = RgbImage::new(src.width() * scale, src.height() * scale);
    for (i, &(cx, cy)) in origins.iter().enumerate() {
        let input = extract_tile(src, cfg, cx, cy);
        let upscaled = upscale(&input);
        blit_central(&mut out, &upscaled, cfg, cx, cy);
        progress((i + 1) as f32 / total);
    }
    out
}

/// Same result as [`upscale_with_tiles`], but tiles are upscaled in parallel
/// across CPU cores. `upscale` must therefore be callable from multiple threads
/// (`Fn + Sync`); `progress` likewise (it is invoked from worker threads as each
/// tile finishes, so its `0.0..=1.0` values may arrive out of order).
pub fn upscale_with_tiles_parallel<F, P>(src: &RgbImage, cfg: &TilingConfig, upscale: F, progress: P) -> RgbImage
where
    F: Fn(&RgbImage) -> RgbImage + Sync,
    P: Fn(f32) + Sync,
{
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let scale = cfg.scale.max(1);
    let origins = tile_origins(src.width(), src.height(), cfg);
    let total = origins.len().max(1);
    let done = AtomicUsize::new(0);

    // Upscale every tile concurrently; collect preserves source order.
    let upscaled: Vec<RgbImage> = origins
        .par_iter()
        .map(|&(cx, cy)| {
            let input = extract_tile(src, cfg, cx, cy);
            let out = upscale(&input);
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            progress(n as f32 / total as f32);
            out
        })
        .collect();

    // Stitch sequentially (cheap memory copies).
    let mut out = RgbImage::new(src.width() * scale, src.height() * scale);
    for (&(cx, cy), tile) in origins.iter().zip(upscaled.iter()) {
        blit_central(&mut out, tile, cfg, cx, cy);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        })
    }

    fn scale4(t: &RgbImage) -> RgbImage {
        let mut o = RgbImage::new(t.width() * 4, t.height() * 4);
        for y in 0..o.height() {
            for x in 0..o.width() {
                o.put_pixel(x, y, *t.get_pixel(x / 4, y / 4));
            }
        }
        o
    }

    #[test]
    fn identity_reconstructs_source_exactly() {
        let src = gradient(50, 37);
        let cfg = TilingConfig { tile_size: 16, overlap: 4, scale: 1 };
        let out = upscale_with_tiles(&src, &cfg, |t| t.clone(), |_| {});
        assert_eq!(out.dimensions(), src.dimensions());
        assert_eq!(out.into_raw(), src.into_raw());
    }

    #[test]
    fn identity_reconstructs_when_smaller_than_tile() {
        let src = gradient(10, 6);
        let cfg = TilingConfig { tile_size: 32, overlap: 8, scale: 1 };
        let out = upscale_with_tiles(&src, &cfg, |t| t.clone(), |_| {});
        assert_eq!(out.dimensions(), src.dimensions());
        assert_eq!(out.into_raw(), src.into_raw());
    }

    #[test]
    fn output_dimensions_scale_correctly() {
        let src = gradient(20, 13);
        let cfg = TilingConfig { tile_size: 8, overlap: 2, scale: 4 };
        let out = upscale_with_tiles(&src, &cfg, scale4, |_| {});
        assert_eq!(out.dimensions(), (80, 52));
    }

    #[test]
    fn progress_reaches_one() {
        let src = gradient(30, 30);
        let cfg = TilingConfig { tile_size: 16, overlap: 4, scale: 1 };
        let mut last = 0.0f32;
        upscale_with_tiles(&src, &cfg, |t| t.clone(), |p| last = p);
        assert!((last - 1.0).abs() < 1e-6, "final progress was {last}");
    }

    #[test]
    fn parallel_matches_sequential_identity() {
        let src = gradient(50, 37);
        let cfg = TilingConfig { tile_size: 16, overlap: 4, scale: 1 };
        let seq = upscale_with_tiles(&src, &cfg, |t| t.clone(), |_| {});
        let par = upscale_with_tiles_parallel(&src, &cfg, |t| t.clone(), |_| {});
        assert_eq!(seq.dimensions(), par.dimensions());
        assert_eq!(seq.into_raw(), par.into_raw());
    }

    #[test]
    fn parallel_matches_sequential_scaled() {
        let src = gradient(37, 41);
        let cfg = TilingConfig { tile_size: 8, overlap: 2, scale: 4 };
        let seq = upscale_with_tiles(&src, &cfg, scale4, |_| {});
        let par = upscale_with_tiles_parallel(&src, &cfg, scale4, |_| {});
        assert_eq!(seq.into_raw(), par.into_raw());
    }

    #[test]
    fn parallel_progress_reaches_one() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let src = gradient(40, 28);
        let cfg = TilingConfig { tile_size: 12, overlap: 3, scale: 1 };
        let max_milli = AtomicU32::new(0);
        upscale_with_tiles_parallel(&src, &cfg, |t| t.clone(), |p| {
            max_milli.fetch_max((p * 1000.0) as u32, Ordering::Relaxed);
        });
        assert_eq!(max_milli.load(Ordering::Relaxed), 1000);
    }
}
