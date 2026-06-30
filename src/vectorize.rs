//! Raster → vector (SVG) tracing, kept GUI-independent like the upscaler core.
//!
//! Wraps the pure-Rust `vtracer` engine (no C dependencies, so it cross-compiles
//! to Windows with the same static-linked toolchain as the rest of the app).
//!
//! Tracing is designed for *flat-color* art — logos, icons, line art, clip-art.
//! Photographs trace into huge, posterized SVGs (thousands of paths), so a
//! `max_dimension` cap lets callers downscale first and keep things tractable.

use std::borrow::Cow;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use image::imageops::FilterType;
use image::RgbaImage;
use resvg::{tiny_skia, usvg};
use vtracer::{ColorImage, Config};

use crate::upscaler::image_io;

/// How colors are handled: full color clustering vs. a single black/white layer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VecColorMode {
    Color,
    Binary,
}

impl VecColorMode {
    pub fn label(self) -> &'static str {
        match self {
            VecColorMode::Color => "컬러",
            VecColorMode::Binary => "흑백",
        }
    }

    fn to_vtracer(self) -> vtracer::ColorMode {
        match self {
            VecColorMode::Color => vtracer::ColorMode::Color,
            VecColorMode::Binary => vtracer::ColorMode::Binary,
        }
    }
}

/// Curve-fitting strategy for the traced outlines.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CurveMode {
    /// Pixel-accurate (stair-stepped) — no smoothing.
    Pixel,
    /// Straight polygon segments.
    Polygon,
    /// Smooth splines (best for organic shapes).
    Spline,
}

impl CurveMode {
    pub fn label(self) -> &'static str {
        match self {
            CurveMode::Pixel => "픽셀 (각짐)",
            CurveMode::Polygon => "다각형",
            CurveMode::Spline => "곡선 (부드러움)",
        }
    }

    fn to_vtracer(self) -> visioncortex::PathSimplifyMode {
        match self {
            CurveMode::Pixel => visioncortex::PathSimplifyMode::None,
            CurveMode::Polygon => visioncortex::PathSimplifyMode::Polygon,
            CurveMode::Spline => visioncortex::PathSimplifyMode::Spline,
        }
    }
}

/// User-facing tracing settings, decoupled from vtracer's `Config` so the GUI
/// and CLI depend on these stable enums instead of the engine's types.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct VectorizeConfig {
    pub color_mode: VecColorMode,
    /// Curve fitting for the outlines.
    pub curve_mode: CurveMode,
    /// Discard color patches smaller than this many pixels (despeckle).
    pub filter_speckle: usize,
    /// Number of significant bits used to quantize colors (1..=8).
    pub color_precision: i32,
    /// Gradient step between adjacent color layers.
    pub layer_difference: i32,
    /// Minimum corner angle in degrees; smaller keeps more corners.
    pub corner_threshold: i32,
    /// Minimum traced segment length (3.5..=10).
    pub length_threshold: f64,
    /// Splice angle threshold in degrees.
    pub splice_threshold: i32,
    /// Curve-fitting iteration budget.
    pub max_iterations: usize,
    /// Decimal places kept in path coordinates.
    pub path_precision: u32,
    /// Cap the longest side before tracing (0 = no cap). Keeps photos sane.
    pub max_dimension: u32,
}

impl Default for VectorizeConfig {
    fn default() -> Self {
        // Borrow vtracer's own defaults for the numeric knobs so we track the
        // engine's tuned values, then layer our own ergonomic choices on top.
        let d = Config::default();
        Self {
            color_mode: VecColorMode::Color,
            curve_mode: CurveMode::Spline,
            filter_speckle: d.filter_speckle,
            color_precision: d.color_precision,
            layer_difference: d.layer_difference,
            corner_threshold: d.corner_threshold,
            length_threshold: d.length_threshold,
            splice_threshold: d.splice_threshold,
            max_iterations: d.max_iterations,
            path_precision: d.path_precision.unwrap_or(2),
            // Tracing scales super-linearly with pixel count; 1000px on the
            // longest side keeps even busy images responsive while staying crisp
            // for the flat-color art this is meant for.
            max_dimension: 1000,
        }
    }
}

impl VectorizeConfig {
    fn to_vtracer(&self) -> Config {
        // Start from defaults so any field we don't surface keeps a sane value.
        let mut c = Config::default();
        c.color_mode = self.color_mode.to_vtracer();
        c.mode = self.curve_mode.to_vtracer();
        c.filter_speckle = self.filter_speckle;
        c.color_precision = self.color_precision;
        c.layer_difference = self.layer_difference;
        c.corner_threshold = self.corner_threshold;
        c.length_threshold = self.length_threshold;
        c.splice_threshold = self.splice_threshold;
        c.max_iterations = self.max_iterations;
        c.path_precision = Some(self.path_precision);
        c
    }
}

/// Downscale so the longest side is at most `max_dim` (0 = leave as-is),
/// borrowing the original when no resize is needed.
fn capped(img: &RgbaImage, max_dim: u32) -> Cow<'_, RgbaImage> {
    let longest = img.width().max(img.height());
    if max_dim == 0 || longest <= max_dim {
        return Cow::Borrowed(img);
    }
    let scale = max_dim as f32 / longest as f32;
    let nw = ((img.width() as f32 * scale).round() as u32).max(1);
    let nh = ((img.height() as f32 * scale).round() as u32).max(1);
    Cow::Owned(image::imageops::resize(img, nw, nh, FilterType::Lanczos3))
}

/// Trace an in-memory RGBA image into an SVG document string.
pub fn vectorize_rgba(img: &RgbaImage, cfg: &VectorizeConfig) -> Result<String> {
    let img = capped(img, cfg.max_dimension);
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return Err(anyhow!("이미지 크기가 0입니다."));
    }
    let ci = ColorImage {
        pixels: img.as_raw().to_vec(),
        width: w,
        height: h,
    };
    let svg = vtracer::convert(ci, cfg.to_vtracer()).map_err(|e| anyhow!("벡터 변환 실패: {e}"))?;
    Ok(svg.to_string())
}

/// Load `input` (any supported raster format), trace it, and write the SVG to
/// `output`.
pub fn vectorize_file(input: &Path, output: &Path, cfg: &VectorizeConfig) -> Result<()> {
    let img = image_io::load_image(input)?.to_rgba8();
    let svg = vectorize_rgba(&img, cfg)?;

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("출력 폴더를 만들 수 없습니다: {}", parent.display()))?;
        }
    }
    std::fs::write(output, svg)
        .with_context(|| format!("SVG를 저장할 수 없습니다: {}", output.display()))?;
    Ok(())
}

/// Rasterize an SVG document into opaque RGBA8 pixels for on-screen preview,
/// scaled so the longest side is at most `max_dim` (0 = native size).
///
/// The canvas is filled white first so the result is fully opaque — vtracer's
/// fills are solid colors, so premultiplied and straight alpha coincide and the
/// bytes can feed egui's `from_rgba_unmultiplied` directly. Returns
/// `(rgba, width, height)`. Uses the pure-Rust `resvg`/`tiny-skia` renderer (no
/// fonts needed — traced output is paths only).
pub fn render_svg_to_rgba(svg: &str, max_dim: u32) -> Result<(Vec<u8>, u32, u32)> {
    let tree = usvg::Tree::from_str(svg, &usvg::Options::default())
        .map_err(|e| anyhow!("SVG 파싱 실패: {e}"))?;
    let size = tree.size();
    let (sw, sh) = (size.width(), size.height());
    if sw <= 0.0 || sh <= 0.0 {
        return Err(anyhow!("SVG 크기가 유효하지 않습니다."));
    }

    let longest = sw.max(sh);
    let scale = if max_dim > 0 && longest > max_dim as f32 {
        max_dim as f32 / longest
    } else {
        1.0
    };
    let w = (sw * scale).ceil().max(1.0) as u32;
    let h = (sh * scale).ceil().max(1.0) as u32;

    let mut pixmap =
        tiny_skia::Pixmap::new(w, h).ok_or_else(|| anyhow!("미리보기 캔버스 생성 실패 ({w}×{h})"))?;
    pixmap.fill(tiny_skia::Color::WHITE);
    resvg::render(
        &tree,
        tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    Ok((pixmap.data().to_vec(), w, h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    /// A red square on white — should trace to at least one filled path.
    fn sample() -> RgbaImage {
        let mut img = RgbaImage::from_pixel(32, 32, Rgba([255, 255, 255, 255]));
        for y in 8..24 {
            for x in 8..24 {
                img.put_pixel(x, y, Rgba([220, 30, 30, 255]));
            }
        }
        img
    }

    #[test]
    fn produces_svg_with_paths() {
        let svg = vectorize_rgba(&sample(), &VectorizeConfig::default()).unwrap();
        assert!(svg.contains("<svg"), "missing <svg root: {svg}");
        assert!(svg.contains("<path"), "missing traced <path: {svg}");
    }

    #[test]
    fn binary_mode_also_works() {
        let cfg = VectorizeConfig {
            color_mode: VecColorMode::Binary,
            ..Default::default()
        };
        let svg = vectorize_rgba(&sample(), &cfg).unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn caps_long_side() {
        let big = RgbaImage::from_pixel(100, 50, Rgba([0, 0, 0, 255]));
        let c = capped(&big, 40);
        assert_eq!(c.width().max(c.height()), 40);
        // Aspect ratio preserved (100:50 -> 40:20).
        assert_eq!((c.width(), c.height()), (40, 20));
    }

    #[test]
    fn no_cap_borrows_original() {
        let small = RgbaImage::from_pixel(10, 10, Rgba([0, 0, 0, 255]));
        let c = capped(&small, 0);
        assert!(matches!(c, Cow::Borrowed(_)));
    }

    #[test]
    fn renders_svg_to_pixels() {
        // Exercise the full trace -> rasterize preview pipeline at runtime.
        let svg = vectorize_rgba(&sample(), &VectorizeConfig::default()).unwrap();
        let (rgba, w, h) = render_svg_to_rgba(&svg, 128).unwrap();
        assert!(w > 0 && h > 0);
        assert_eq!(rgba.len(), (w * h * 4) as usize);
        // The traced shapes must paint some non-white, fully-opaque pixels.
        let painted = rgba
            .chunks_exact(4)
            .any(|p| p[3] == 255 && (p[0] < 240 || p[1] < 240 || p[2] < 240));
        assert!(painted, "expected traced shapes to paint non-white pixels");
    }
}
