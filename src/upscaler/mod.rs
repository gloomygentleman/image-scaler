//! Core upscaling logic, kept independent of the GUI so it can be unit-tested.

pub mod download;
pub mod engine;
pub mod image_io;
pub mod model;
#[cfg(feature = "gpu")]
pub mod model_ort;
pub mod pipeline;
pub mod sizing;
pub mod tiling;

use std::path::{Path, PathBuf};

use image_io::OutputFormat;

/// Decide the output path and format for one input file.
///
/// When `keep_input_format` is set and the input's extension is one we can
/// encode, that format/extension is reused; otherwise we fall back to the
/// user's chosen `format`. `suffix` is appended to the file stem (e.g.
/// `cat` + `_x4` -> `cat_x4.png`).
pub fn resolve_output(
    input: &Path,
    out_dir: &Path,
    format: OutputFormat,
    keep_input_format: bool,
    suffix: &str,
) -> (PathBuf, OutputFormat) {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".to_string());

    let (fmt, ext) = if keep_input_format {
        match input
            .extension()
            .and_then(|e| OutputFormat::from_extension(&e.to_string_lossy()))
        {
            // Reuse the input's extension verbatim so e.g. `.jpeg` stays `.jpeg`.
            Some(f) => (
                f,
                input
                    .extension()
                    .map(|e| e.to_string_lossy().into_owned())
                    .unwrap_or_else(|| f.extension().to_string()),
            ),
            None => (format, format.extension().to_string()),
        }
    } else {
        (format, format.extension().to_string())
    };

    let file_name = format!("{stem}{suffix}.{ext}");
    (out_dir.join(file_name), fmt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_chosen_format_by_default() {
        let (path, fmt) = resolve_output(
            Path::new("/in/cat.jpg"),
            Path::new("/out"),
            OutputFormat::Png,
            false,
            "_x4",
        );
        assert_eq!(fmt, OutputFormat::Png);
        assert_eq!(path, PathBuf::from("/out/cat_x4.png"));
    }

    #[test]
    fn keeps_input_format_when_encodable() {
        let (path, fmt) = resolve_output(
            Path::new("/in/cat.jpeg"),
            Path::new("/out"),
            OutputFormat::Png,
            true,
            "",
        );
        assert_eq!(fmt, OutputFormat::Jpeg);
        assert_eq!(path, PathBuf::from("/out/cat.jpeg"));
    }

    #[test]
    fn falls_back_when_input_format_not_encodable() {
        // GIF input can be decoded but not encoded here -> fall back to chosen.
        let (path, fmt) = resolve_output(
            Path::new("/in/anim.gif"),
            Path::new("/out"),
            OutputFormat::WebP,
            true,
            "_up",
        );
        assert_eq!(fmt, OutputFormat::WebP);
        assert_eq!(path, PathBuf::from("/out/anim_up.webp"));
    }
}
