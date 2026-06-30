//! AI Image Upscaler — CPU-only super-resolution desktop app.
//!
//! Run with `cargo run --release`. Pick input images, an output folder, and an
//! ONNX super-resolution model (see README), then start.

// Hide the console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod fonts;
mod theme;
mod upscaler;
mod vectorize;

use std::path::PathBuf;
use std::process::ExitCode;

use eframe::egui;

use std::sync::atomic::AtomicBool;

use upscaler::engine::Engine;
use upscaler::image_io::OutputFormat;
use upscaler::sizing::TargetSize;
use upscaler::{download, pipeline, resolve_output};
use vectorize::{CurveMode, VecColorMode, VectorizeConfig};

fn main() -> ExitCode {
    // Headless modes for automation / verification:
    //   image-upscaler --cli <model.onnx> <scale> <input> <output> [tile_size] [tw th] [--gpu]
    //   image-upscaler --vectorize <input> <output.svg> [color|binary] [pixel|polygon|spline] [max_dim]
    //   image-upscaler --download [model_index]
    let args: Vec<String> = std::env::args().collect();
    let code: u8 = if args.get(1).map(String::as_str) == Some("--cli") {
        run_or_report(run_cli(&args[2..]))
    } else if args.get(1).map(String::as_str) == Some("--vectorize") {
        run_or_report(run_vectorize(&args[2..]))
    } else if args.get(1).map(String::as_str) == Some("--download") {
        run_or_report(run_download(&args[2..]))
    } else {
        match run_gui() {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("error: {e:#}");
                1
            }
        }
    };

    // ONNX Runtime can abort during C++ static destructors at normal process
    // teardown — *after* all work has finished and output has been written. In
    // GPU builds we bypass those destructors with `_exit` so the process ends
    // cleanly (flushing first); without the `gpu` feature there is no such issue.
    #[cfg(feature = "gpu")]
    {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        extern "C" {
            fn _exit(code: i32) -> !;
        }
        unsafe { _exit(code as i32) }
    }
    #[cfg(not(feature = "gpu"))]
    ExitCode::from(code)
}

/// Map a headless command's result to a process exit code, printing errors.
fn run_or_report(result: anyhow::Result<()>) -> u8 {
    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e:#}");
            1
        }
    }
}

/// Headless single-file upscale, used for scripting and end-to-end tests.
fn run_cli(args: &[String]) -> anyhow::Result<()> {
    // `--gpu` may appear anywhere; strip it from the positional arguments.
    let gpu = args.iter().any(|a| a == "--gpu");
    let pos: Vec<String> = args.iter().filter(|a| a.as_str() != "--gpu").cloned().collect();
    let args = &pos[..];

    if args.len() < 4 {
        anyhow::bail!(
            "usage: image-upscaler --cli <model.onnx> <scale> <input> <output> \
             [tile_size] [target_w target_h] [--gpu]"
        );
    }
    let model = PathBuf::from(&args[0]);
    let scale: u32 = args[1].parse().map_err(|_| anyhow::anyhow!("invalid scale: {}", args[1]))?;
    let input = PathBuf::from(&args[2]);
    let output = PathBuf::from(&args[3]);
    let tile_size: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(128);

    let out_dir = output.parent().map(PathBuf::from).unwrap_or_default();
    let (_path, format) = resolve_output(&output, &out_dir, OutputFormat::Png, true, "");

    eprintln!("loading model {} (x{scale}, tile {tile_size}, gpu={gpu})...", model.display());
    let upscaler = Engine::load(&model, tile_size, scale, gpu)?;
    eprintln!("backend: {}", upscaler.backend_name());

    // Optional explicit target size: `... [tile_size] <target_w> <target_h>`.
    // Without it, the model's native scale is the target (identity resample).
    let target = match (args.get(5).and_then(|s| s.parse().ok()), args.get(6).and_then(|s| s.parse().ok())) {
        (Some(w), Some(h)) => TargetSize::Exact { w, h },
        _ => TargetSize::Scale(scale as f32),
    };

    let mut last = -1i32;
    pipeline::process(&upscaler, &input, &output, target, format, 90, |p| {
        let pct = (p * 100.0) as i32;
        if pct / 10 != last / 10 {
            eprint!("\rupscaling... {pct}%");
            last = pct;
        }
    })?;
    eprintln!("\rdone. wrote {}", output.display());
    Ok(())
}

/// Headless raster → vector (SVG) tracing, for scripting and end-to-end tests.
fn run_vectorize(args: &[String]) -> anyhow::Result<()> {
    if args.len() < 2 {
        anyhow::bail!(
            "usage: image-upscaler --vectorize <input> <output.svg> \
             [color|binary] [pixel|polygon|spline] [max_dim]"
        );
    }
    let input = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);

    let mut cfg = VectorizeConfig::default();
    if let Some(m) = args.get(2) {
        cfg.color_mode = match m.as_str() {
            "color" => VecColorMode::Color,
            "binary" => VecColorMode::Binary,
            other => anyhow::bail!("invalid color mode: {other} (color|binary)"),
        };
    }
    if let Some(m) = args.get(3) {
        cfg.curve_mode = match m.as_str() {
            "pixel" => CurveMode::Pixel,
            "polygon" => CurveMode::Polygon,
            "spline" => CurveMode::Spline,
            other => anyhow::bail!("invalid curve mode: {other} (pixel|polygon|spline)"),
        };
    }
    if let Some(d) = args.get(4) {
        cfg.max_dimension = d.parse().map_err(|_| anyhow::anyhow!("invalid max_dim: {d}"))?;
    }

    eprintln!("vectorizing {} -> {}...", input.display(), output.display());
    vectorize::vectorize_file(&input, &output, &cfg)?;
    let bytes = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    eprintln!("done. wrote {} ({})", output.display(), download::human_bytes(bytes));
    Ok(())
}

/// Headless model download into the per-user cache.
fn run_download(args: &[String]) -> anyhow::Result<()> {
    let idx: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(0);
    let spec = download::MODELS
        .get(idx)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("invalid model index {idx} (have {})", download::MODELS.len()))?;

    eprintln!("downloading \"{}\" -> {}", spec.label, download::model_path(&spec).display());
    let cancel = AtomicBool::new(false);
    let mut last = -1i64;
    let path = download::ensure_model(&spec, &cancel, |downloaded, total| {
        if let Some(total) = total {
            let pct = (downloaded * 100 / total.max(1)) as i64;
            if pct / 5 != last / 5 {
                eprint!(
                    "\r{} / {} ({pct}%)   ",
                    download::human_bytes(downloaded),
                    download::human_bytes(total)
                );
                last = pct;
            }
        }
    })?;
    eprintln!("\rdone. cached at {}", path.display());
    Ok(())
}

fn run_gui() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([820.0, 560.0])
            .with_min_inner_size([560.0, 420.0]),
        ..Default::default()
    };

    eframe::run_native(
        "AI Image Upscaler",
        options,
        Box::new(|cc| {
            // Embed Korean fonts so Hangul renders, then apply the visual theme.
            fonts::install_korean_fonts(&cc.egui_ctx);
            theme::apply(&cc.egui_ctx);
            Ok(Box::new(app::UpscalerApp::default()) as Box<dyn eframe::App>)
        }),
    )
}
