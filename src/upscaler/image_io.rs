//! Multi-format image loading and saving.
//!
//! Input: any format the `image` crate can decode (png, jpg, bmp, webp, tiff, gif, ...).
//! Output: a user-selectable subset, including a lossy format with quality control.

use std::path::Path;

use anyhow::{Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageFormat};

/// Output container formats the user can choose from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OutputFormat {
    Png,
    Jpeg,
    WebP,
    Bmp,
    Tiff,
}

impl OutputFormat {
    /// All selectable formats, in display order.
    pub const ALL: [OutputFormat; 5] = [
        OutputFormat::Png,
        OutputFormat::Jpeg,
        OutputFormat::WebP,
        OutputFormat::Bmp,
        OutputFormat::Tiff,
    ];

    /// Human-readable label for the GUI.
    pub fn label(self) -> &'static str {
        match self {
            OutputFormat::Png => "PNG (lossless)",
            OutputFormat::Jpeg => "JPEG (lossy)",
            OutputFormat::WebP => "WebP (lossless)",
            OutputFormat::Bmp => "BMP (lossless)",
            OutputFormat::Tiff => "TIFF (lossless)",
        }
    }

    /// Canonical file extension (without the dot).
    pub fn extension(self) -> &'static str {
        match self {
            OutputFormat::Png => "png",
            OutputFormat::Jpeg => "jpg",
            OutputFormat::WebP => "webp",
            OutputFormat::Bmp => "bmp",
            OutputFormat::Tiff => "tiff",
        }
    }

    /// Map a file extension to an output format, if we can encode it.
    pub fn from_extension(ext: &str) -> Option<OutputFormat> {
        match ext.to_ascii_lowercase().as_str() {
            "png" => Some(OutputFormat::Png),
            "jpg" | "jpeg" => Some(OutputFormat::Jpeg),
            "webp" => Some(OutputFormat::WebP),
            "bmp" => Some(OutputFormat::Bmp),
            "tif" | "tiff" => Some(OutputFormat::Tiff),
            _ => None,
        }
    }

    /// Whether the format is lossy (and therefore uses the quality setting).
    pub fn is_lossy(self) -> bool {
        matches!(self, OutputFormat::Jpeg)
    }
}

/// Extensions accepted on input, for the file-picker filter.
pub const SUPPORTED_INPUT_EXTENSIONS: &[&str] =
    &["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff", "gif"];

/// Load an image from disk in any supported format.
pub fn load_image(path: &Path) -> Result<DynamicImage> {
    image::open(path).with_context(|| format!("이미지를 불러올 수 없습니다: {}", path.display()))
}

/// Save an image to disk in the requested format.
///
/// `quality` is in `1..=100` and only affects lossy formats (JPEG). The other
/// formats are written losslessly. JPEG cannot store transparency, so an alpha
/// channel is flattened to RGB for that format only.
pub fn save_image(img: &DynamicImage, path: &Path, format: OutputFormat, quality: u8) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("출력 폴더를 만들 수 없습니다: {}", parent.display()))?;
        }
    }

    let ctx = || format!("이미지를 저장할 수 없습니다: {}", path.display());

    match format {
        OutputFormat::Jpeg => {
            let file = std::fs::File::create(path).with_context(ctx)?;
            let mut writer = std::io::BufWriter::new(file);
            let mut encoder = JpegEncoder::new_with_quality(&mut writer, quality.clamp(1, 100));
            encoder
                .encode_image(&img.to_rgb8())
                .with_context(ctx)?;
        }
        OutputFormat::Png => img.save_with_format(path, ImageFormat::Png).with_context(ctx)?,
        OutputFormat::WebP => img.save_with_format(path, ImageFormat::WebP).with_context(ctx)?,
        OutputFormat::Bmp => img.save_with_format(path, ImageFormat::Bmp).with_context(ctx)?,
        OutputFormat::Tiff => img.save_with_format(path, ImageFormat::Tiff).with_context(ctx)?,
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_roundtrip() {
        for fmt in OutputFormat::ALL {
            let ext = fmt.extension();
            assert_eq!(OutputFormat::from_extension(ext), Some(fmt));
        }
    }

    #[test]
    fn jpeg_aliases_map_to_jpeg() {
        assert_eq!(OutputFormat::from_extension("JPEG"), Some(OutputFormat::Jpeg));
        assert_eq!(OutputFormat::from_extension("jpg"), Some(OutputFormat::Jpeg));
    }

    #[test]
    fn only_jpeg_is_lossy() {
        for fmt in OutputFormat::ALL {
            assert_eq!(fmt.is_lossy(), fmt == OutputFormat::Jpeg);
        }
    }
}
