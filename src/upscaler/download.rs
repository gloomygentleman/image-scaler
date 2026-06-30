//! Optional model auto-download with on-disk caching.
//!
//! Keeps the app usable out of the box: a curated super-resolution model can be
//! fetched on demand into a per-user cache directory and reused thereafter.
//! Downloads are streamed to a `.part` file and atomically renamed on success,
//! so an interrupted download never leaves a half-written model in place.

use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{bail, Context, Result};

/// A downloadable, pre-trained ONNX super-resolution model.
#[derive(Clone, Copy)]
pub struct ModelSpec {
    pub label: &'static str,
    pub file_name: &'static str,
    pub url: &'static str,
    pub scale: u32,
    pub approx_mb: u32,
}

/// Curated models known to satisfy the app's I/O contract
/// (NCHW RGB float32 in `[0,1]`, dynamic H/W). Verified to load and run.
pub const MODELS: &[ModelSpec] = &[ModelSpec {
    label: "Real-ESRGAN x4plus (범용, 4x)",
    file_name: "realesrgan-x4plus.onnx",
    url: "https://huggingface.co/spaces/Wuvin/Unique3D/resolve/main/ckpt/realesrgan-x4.onnx",
    scale: 4,
    approx_mb: 64,
}];

/// Per-user cache directory for downloaded models.
pub fn cache_dir() -> PathBuf {
    #[cfg(windows)]
    {
        if let Ok(d) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(d).join("image-upscaler").join("models");
        }
    }
    #[cfg(not(windows))]
    {
        if let Ok(d) = std::env::var("XDG_CACHE_HOME") {
            return PathBuf::from(d).join("image-upscaler").join("models");
        }
        if let Ok(h) = std::env::var("HOME") {
            return PathBuf::from(h)
                .join(".cache")
                .join("image-upscaler")
                .join("models");
        }
    }
    PathBuf::from("models")
}

/// Where a given model is (or would be) cached.
pub fn model_path(spec: &ModelSpec) -> PathBuf {
    cache_dir().join(spec.file_name)
}

/// Whether the model is already present in the cache.
pub fn is_downloaded(spec: &ModelSpec) -> bool {
    model_path(spec).is_file()
}

/// Ensure `spec` is cached locally, downloading it if needed.
///
/// `progress` receives `(downloaded_bytes, total_bytes_opt)`. Honours `cancel`.
pub fn ensure_model(
    spec: &ModelSpec,
    cancel: &AtomicBool,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<PathBuf> {
    let dest = model_path(spec);
    if dest.is_file() {
        return Ok(dest);
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("캐시 폴더를 만들 수 없습니다: {}", parent.display()))?;
    }

    let resp = ureq::get(spec.url)
        .call()
        .with_context(|| format!("다운로드에 실패했습니다: {}", spec.url))?;
    let total = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok());

    let tmp = dest.with_extension("part");
    let mut file = fs::File::create(&tmp)
        .with_context(|| format!("임시 파일을 만들 수 없습니다: {}", tmp.display()))?;
    let mut reader = resp.into_reader();
    let mut buf = [0u8; 64 * 1024];
    let mut downloaded: u64 = 0;

    loop {
        if cancel.load(Ordering::SeqCst) {
            drop(file);
            let _ = fs::remove_file(&tmp);
            bail!("다운로드가 취소되었습니다");
        }
        let n = reader.read(&mut buf).context("다운로드 중 오류가 발생했습니다")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("파일 쓰기에 실패했습니다")?;
        downloaded += n as u64;
        progress(downloaded, total);
    }

    file.flush().ok();
    drop(file);
    fs::rename(&tmp, &dest)
        .with_context(|| format!("모델 저장에 실패했습니다: {}", dest.display()))?;
    Ok(dest)
}

/// Format a byte count as a short human-readable string.
pub fn human_bytes(n: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    let n = n as f64;
    if n >= MB {
        format!("{:.1} MB", n / MB)
    } else if n >= KB {
        format!("{:.0} KB", n / KB)
    } else {
        format!("{n:.0} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_bytes_scales_units() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2048), "2 KB");
        assert_eq!(human_bytes(64 * 1024 * 1024), "64.0 MB");
    }

    #[test]
    fn model_path_lives_under_cache_dir() {
        let spec = MODELS[0];
        assert!(model_path(&spec).starts_with(cache_dir()));
        assert!(model_path(&spec).ends_with(spec.file_name));
    }

    #[test]
    fn curated_models_are_well_formed() {
        for spec in MODELS {
            assert!(spec.url.starts_with("https://"), "model URL must be https");
            assert!(spec.scale >= 2);
            assert!(spec.file_name.ends_with(".onnx"));
        }
    }
}
