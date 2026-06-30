//! egui desktop front-end.
//!
//! All heavy work (model loading + inference) runs on a background thread so the
//! UI stays responsive; the worker reports progress and results back over a
//! channel and can be cancelled cooperatively.
//!
//! Layout follows the actual workflow as a numbered sequence — ① images,
//! ② target size, ③ output, ④ model — with one always-visible primary action.
//! The signature element is the draggable before/after comparison in the
//! preview pane, which makes the resolution gain tangible.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::thread;

use eframe::egui::{
    self, Align2, Color32, FontId, Margin, Pos2, Rect, Rounding, Sense, Stroke, Vec2,
};

use crate::theme;
use crate::upscaler::engine::Engine;
use crate::upscaler::image_io::{OutputFormat, SUPPORTED_INPUT_EXTENSIONS};
use crate::upscaler::sizing::TargetSize;
use crate::upscaler::{download, pipeline, resolve_output};
use crate::vectorize::{self, CurveMode, VecColorMode, VectorizeConfig};

/// The two top-level workflows the app offers.
#[derive(Clone, Copy, PartialEq)]
enum AppMode {
    /// AI super-resolution (raster → larger raster).
    Upscale,
    /// Image tracing (raster → vector SVG).
    Vectorize,
}

/// Whether the user sizes the output by a factor or by explicit dimensions.
#[derive(Clone, Copy, PartialEq)]
enum SizeMode {
    Scale,
    Dimensions,
}

/// Messages sent from the upscaling worker thread back to the UI.
enum WorkerMsg {
    Progress(f32),
    Log(String),
    FileDone { output: PathBuf },
    Finished,
}

/// Messages sent from the model-download worker thread back to the UI.
enum DlMsg {
    Progress { downloaded: u64, total: Option<u64> },
    Done { path: PathBuf, scale: u32 },
    Failed(String),
}

pub struct UpscalerApp {
    /// Which workflow is active (upscale vs. vectorize).
    mode: AppMode,
    /// Tracing settings for the vectorize workflow.
    vec_cfg: VectorizeConfig,

    // Inputs / configuration.
    input_files: Vec<PathBuf>,
    output_dir: Option<PathBuf>,
    model_path: Option<PathBuf>,
    output_format: OutputFormat,
    keep_input_format: bool,
    tile_size: u32,
    quality: u8,
    suffix: String,

    // Target size.
    size_mode: SizeMode,
    target_scale: f32,
    target_w: u32,
    target_h: u32,
    lock_aspect: bool,
    /// Dimensions of the first selected image, for the live size readout.
    src_dims: Option<(u32, u32)>,

    // Model: its native scale (must match the model file) + auto-download.
    model_scale: u32,
    /// Prefer the GPU (ONNX Runtime) backend when available (gpu builds only).
    use_gpu: bool,
    selected_model_idx: usize,
    dl_rx: Option<Receiver<DlMsg>>,

    // Runtime state.
    status: String,
    progress: f32,
    running: bool,
    cancel_flag: Arc<AtomicBool>,
    rx: Option<Receiver<WorkerMsg>>,

    // Preview.
    input_preview: Option<egui::TextureHandle>,
    output_preview: Option<egui::TextureHandle>,
    preview_source: Option<PathBuf>,
    last_output: Option<PathBuf>,
    /// Split position (0..1) of the before/after comparison slider.
    compare_split: f32,
}

impl Default for UpscalerApp {
    fn default() -> Self {
        // Pre-select the curated model if it's already cached, so the user
        // doesn't have to choose a model file on every launch.
        let spec = download::MODELS[0];
        let cached = download::is_downloaded(&spec);
        let model_path = cached.then(|| download::model_path(&spec));
        let model_scale = if cached { spec.scale } else { 4 };
        let status = if cached {
            format!("모델 준비됨: {} · 이미지와 출력 폴더만 고르면 시작!", spec.label)
        } else {
            "이미지와 모델을 준비하면 시작할 수 있어요.".to_string()
        };

        Self {
            mode: AppMode::Upscale,
            vec_cfg: VectorizeConfig::default(),
            input_files: Vec::new(),
            output_dir: None,
            model_path,
            output_format: OutputFormat::Png,
            keep_input_format: false,
            tile_size: 128,
            quality: 90,
            suffix: "_upscaled".to_string(),
            size_mode: SizeMode::Scale,
            target_scale: 4.0,
            target_w: 1920,
            target_h: 1080,
            lock_aspect: true,
            src_dims: None,
            model_scale,
            use_gpu: cfg!(feature = "gpu"),
            selected_model_idx: 0,
            dl_rx: None,
            status,
            progress: 0.0,
            running: false,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            rx: None,
            input_preview: None,
            output_preview: None,
            preview_source: None,
            last_output: None,
            compare_split: 0.5,
        }
    }
}

impl eframe::App for UpscalerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ui(ctx);
    }
}

impl UpscalerApp {
    /// Render the whole UI for one frame. Split out from `update` so the full
    /// layout can be exercised headlessly in tests (no `eframe::Frame` needed).
    fn ui(&mut self, ctx: &egui::Context) {
        self.drain_worker_messages(ctx);
        self.ensure_input_preview(ctx);

        self.header(ctx);
        self.status_bar(ctx);

        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(316.0)
            .width_range(300.0..=470.0)
            .frame(egui::Frame::none().fill(theme::BG_PANEL))
            .show(ctx, |ui| {
                // Pin the primary action to the bottom of the panel.
                egui::TopBottomPanel::bottom("actions")
                    .frame(
                        egui::Frame::none()
                            .fill(theme::BG_PANEL)
                            .inner_margin(Margin::symmetric(11.0, 9.0)),
                    )
                    .show_separator_line(false)
                    .show_inside(ui, |ui| self.action_row(ui, ctx));

                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::Frame::none()
                        .inner_margin(Margin::symmetric(10.0, 9.0))
                        .show(ui, |ui| self.controls_ui(ui, ctx));
                });
            });

        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(theme::BG_WINDOW)
                    .inner_margin(Margin::same(13.0)),
            )
            .show(ctx, |ui| self.preview_ui(ui));
    }
}

impl UpscalerApp {
    // ---- Top-level chrome ---------------------------------------------------

    fn header(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("header")
            .frame(
                egui::Frame::none()
                    .fill(theme::BG_PANEL)
                    .inner_margin(Margin::symmetric(13.0, 8.0)),
            )
            .show_separator_line(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // Logo mark: an accent tile with an "enlarge" arrow.
                    let (mark, _) = ui.allocate_exact_size(Vec2::splat(22.0), Sense::hover());
                    ui.painter().rect_filled(mark, Rounding::same(6.0), theme::ACCENT);
                    ui.painter().text(
                        mark.center(),
                        Align2::CENTER_CENTER,
                        "⬆",
                        FontId::proportional(13.0),
                        theme::ON_ACCENT,
                    );
                    ui.add_space(8.0);
                    let (title, subtitle) = match self.mode {
                        AppMode::Upscale => (
                            "이미지 업스케일러",
                            "AI 초해상도 · CPU 전용 · 단일 실행 파일",
                        ),
                        AppMode::Vectorize => (
                            "이미지 벡터 변환",
                            "래스터 → SVG 트레이싱 · 로고·아이콘·라인아트에 최적",
                        ),
                    };
                    ui.vertical(|ui| {
                        ui.label(theme::title(title, 16.0));
                        ui.label(
                            egui::RichText::new(subtitle)
                                .size(10.5)
                                .color(theme::TEXT_MUTED),
                        );
                    });

                    // Right-aligned mode switch.
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        self.mode_switch(ui);
                    });
                });
            });
    }

    /// Segmented [업스케일 | 벡터 변환] toggle. Switching modes clears the
    /// mode-specific result preview so the two workflows don't bleed into each
    /// other. Disabled while a job is running.
    fn mode_switch(&mut self, ui: &mut egui::Ui) {
        ui.add_enabled_ui(!self.running, |ui| {
            ui.horizontal(|ui| {
                // Lay out left-to-right inside this right-aligned region.
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    let mut pick = |ui: &mut egui::Ui, mode: AppMode, label: &str| {
                        if ui
                            .add_sized(
                                [88.0, 26.0],
                                egui::SelectableLabel::new(self.mode == mode, label),
                            )
                            .clicked()
                            && self.mode != mode
                        {
                            self.mode = mode;
                            self.output_preview = None;
                            self.last_output = None;
                            self.progress = 0.0;
                        }
                    };
                    pick(ui, AppMode::Upscale, "업스케일");
                    pick(ui, AppMode::Vectorize, "벡터 변환");
                });
            });
        });
    }

    fn status_bar(&self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("statusbar")
            .frame(
                egui::Frame::none()
                    .fill(theme::BG_PANEL)
                    .inner_margin(Margin::symmetric(13.0, 7.0)),
            )
            .show_separator_line(false)
            .show(ctx, |ui| {
                ui.add(
                    egui::ProgressBar::new(self.progress)
                        .fill(theme::ACCENT)
                        .rounding(Rounding::same(5.0))
                        .desired_width(ui.available_width()),
                );
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(&self.status)
                        .color(theme::TEXT_MUTED)
                        .size(11.0),
                );
            });
    }

    // ---- Left control panel -------------------------------------------------

    fn controls_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let busy = self.running;

        // ① Images / output folder.
        card(ui, "1", "이미지", |ui| {
            ui.add_enabled_ui(!busy, |ui| {
                if ui
                    .add(wide_button(ui, "🖼  이미지 선택"))
                    .on_hover_text("여러 장을 한 번에 선택할 수 있어요")
                    .clicked()
                {
                    if let Some(files) = rfd::FileDialog::new()
                        .add_filter("이미지", SUPPORTED_INPUT_EXTENSIONS)
                        .pick_files()
                    {
                        self.input_files = files;
                        self.preview_source = None;
                        self.output_preview = None;
                        self.refresh_source_dims();
                        self.status = format!("이미지 {}장을 선택했어요.", self.input_files.len());
                    }
                }

                let count = self.input_files.len();
                let (txt, col) = if count > 0 {
                    (format!("선택됨: {count}장"), theme::ACCENT)
                } else {
                    ("아직 선택한 이미지가 없어요".to_string(), theme::TEXT_MUTED)
                };
                ui.label(egui::RichText::new(txt).size(12.5).color(col));

                ui.add_space(2.0);
                if ui.add(wide_button(ui, "📁  출력 폴더 선택")).clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.output_dir = Some(dir);
                    }
                }
                ui.label(
                    egui::RichText::new(format!("저장 위치: {}", path_hint(&self.output_dir)))
                        .size(12.0)
                        .color(theme::TEXT_MUTED),
                );
            });
        });

        match self.mode {
            AppMode::Upscale => self.upscale_controls(ui, ctx),
            AppMode::Vectorize => self.vectorize_controls(ui),
        }
    }

    /// Cards ②–④ for the upscale workflow: target size, output format, model.
    fn upscale_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let busy = self.running;

        // ② Target size — the star of the panel.
        card(ui, "2", "목표 크기", |ui| {
            ui.add_enabled_ui(!busy, |ui| {
                self.size_controls(ui);
            });
        });

        // ③ Output format.
        card(ui, "3", "출력", |ui| {
            ui.add_enabled_ui(!busy, |ui| {
                ui.checkbox(&mut self.keep_input_format, "입력과 동일한 포맷으로 저장");

                ui.add_enabled_ui(!self.keep_input_format, |ui| {
                    field_label(ui, "포맷");
                    let w = (ui.available_width() - 6.0).max(80.0);
                    egui::ComboBox::from_id_source("output_format")
                        .selected_text(self.output_format.label())
                        .width(w)
                        .truncate()
                        .show_ui(ui, |ui| {
                            for fmt in OutputFormat::ALL {
                                ui.selectable_value(&mut self.output_format, fmt, fmt.label());
                            }
                        });
                });

                let lossy = self.output_format.is_lossy() && !self.keep_input_format;
                ui.add_enabled_ui(lossy, |ui| {
                    field_label(ui, "JPEG 품질");
                    ui.add(egui::Slider::new(&mut self.quality, 1..=100));
                });

                ui.horizontal(|ui| {
                    ui.label("파일명 접미사");
                    ui.text_edit_singleline(&mut self.suffix);
                });
            });
        });

        // ④ Model (advanced).
        card(ui, "4", "모델", |ui| {
            ui.add_enabled_ui(!busy, |ui| {
                if ui.add(wide_button(ui, "🧠  ONNX 모델 선택 (.onnx)")).clicked() {
                    if let Some(file) = rfd::FileDialog::new()
                        .add_filter("ONNX 모델", &["onnx"])
                        .pick_file()
                    {
                        self.model_path = Some(file);
                    }
                }
                let model_name = self
                    .model_path
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().to_string());
                let (txt, col) = match &model_name {
                    Some(n) => (format!("모델: {n}"), theme::GOOD),
                    None => ("모델이 아직 없어요 — 아래에서 받을 수 있어요".to_string(), theme::TEXT_MUTED),
                };
                ui.label(egui::RichText::new(txt).size(12.0).color(col));

                ui.add_space(4.0);
                self.download_section(ui, ctx);

                ui.add_space(6.0);
                egui::CollapsingHeader::new("고급 설정")
                    .default_open(false)
                    .show(ui, |ui| {
                        field_label(ui, "모델 배율 (모델과 일치)");
                        let w = (ui.available_width() - 6.0).max(80.0);
                        egui::ComboBox::from_id_source("model_scale")
                            .selected_text(format!("{}×", self.model_scale))
                            .width(w)
                            .show_ui(ui, |ui| {
                                for s in [2u32, 3, 4] {
                                    ui.selectable_value(&mut self.model_scale, s, format!("{s}×"));
                                }
                            });
                        ui.add_space(4.0);
                        field_label(ui, "타일 크기");
                        let w = (ui.available_width() - 6.0).max(80.0);
                        egui::ComboBox::from_id_source("tile_size")
                            .selected_text(format!("{} px", self.tile_size))
                            .width(w)
                            .show_ui(ui, |ui| {
                                for t in [64u32, 128, 192, 256] {
                                    ui.selectable_value(&mut self.tile_size, t, format!("{t} px"));
                                }
                            });
                        ui.add_space(6.0);
                        #[cfg(feature = "gpu")]
                        {
                            ui.checkbox(
                                &mut self.use_gpu,
                                "GPU 가속 사용 (CoreML/DirectML, 실패 시 CPU)",
                            );
                        }
                        #[cfg(not(feature = "gpu"))]
                        {
                            ui.label(
                                egui::RichText::new(
                                    "GPU 가속은 'gpu' 기능으로 빌드할 때 켜집니다 \
                                     (cargo run --release --features gpu).",
                                )
                                .size(11.0)
                                .color(theme::TEXT_MUTED),
                            );
                        }

                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(
                                "모델은 자기 고유 배율로 한 번 처리한 뒤, 목표 크기에 맞춰 \
                                 고품질로 다시 샘플링합니다. CPU는 타일을 코어별로 병렬 처리합니다.",
                            )
                            .size(11.5)
                            .color(theme::TEXT_MUTED),
                        );
                    });
            });
        });
    }

    /// Card ② for the vectorize workflow: tracing options (no model needed).
    fn vectorize_controls(&mut self, ui: &mut egui::Ui) {
        let busy = self.running;

        card(ui, "2", "변환 옵션", |ui| {
            ui.add_enabled_ui(!busy, |ui| {
                // Color vs. single black/white layer.
                field_label(ui, "색상");
                ui.horizontal(|ui| {
                    let w = (ui.available_width() - ui.spacing().item_spacing.x) / 2.0;
                    for m in [VecColorMode::Color, VecColorMode::Binary] {
                        if ui
                            .add_sized(
                                [w, 26.0],
                                egui::SelectableLabel::new(self.vec_cfg.color_mode == m, m.label()),
                            )
                            .clicked()
                        {
                            self.vec_cfg.color_mode = m;
                        }
                    }
                });

                ui.add_space(6.0);
                field_label(ui, "곡선 방식");
                let w = (ui.available_width() - 6.0).max(80.0);
                egui::ComboBox::from_id_source("vec_curve")
                    .selected_text(self.vec_cfg.curve_mode.label())
                    .width(w)
                    .show_ui(ui, |ui| {
                        for m in [CurveMode::Spline, CurveMode::Polygon, CurveMode::Pixel] {
                            ui.selectable_value(&mut self.vec_cfg.curve_mode, m, m.label());
                        }
                    });

                ui.add_space(6.0);
                field_label(ui, "잡티 제거 (작은 점 무시)");
                ui.add(egui::Slider::new(&mut self.vec_cfg.filter_speckle, 0..=16));

                ui.add_enabled_ui(self.vec_cfg.color_mode == VecColorMode::Color, |ui| {
                    ui.add_space(4.0);
                    field_label(ui, "색상 정밀도 (비트)");
                    ui.add(egui::Slider::new(&mut self.vec_cfg.color_precision, 1..=8));
                });

                ui.add_space(6.0);
                field_label(ui, "최대 처리 크기 (클수록 정밀·느림)");
                let w = (ui.available_width() - 6.0).max(80.0);
                let cur_label = if self.vec_cfg.max_dimension == 0 {
                    "원본 그대로".to_string()
                } else {
                    format!("{} px", self.vec_cfg.max_dimension)
                };
                egui::ComboBox::from_id_source("vec_maxdim")
                    .selected_text(cur_label)
                    .width(w)
                    .show_ui(ui, |ui| {
                        for d in [500u32, 1000, 2000] {
                            ui.selectable_value(
                                &mut self.vec_cfg.max_dimension,
                                d,
                                format!("{d} px"),
                            );
                        }
                        ui.selectable_value(&mut self.vec_cfg.max_dimension, 0, "원본 그대로");
                    });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("파일명 접미사");
                    ui.text_edit_singleline(&mut self.suffix);
                });
                ui.label(
                    egui::RichText::new("결과는 .svg(벡터)로 저장됩니다.")
                        .size(11.5)
                        .color(theme::TEXT_MUTED),
                );

                ui.add_space(6.0);
                egui::CollapsingHeader::new("고급 설정")
                    .default_open(false)
                    .show(ui, |ui| {
                        field_label(ui, "코너 임계값 (작을수록 각진 코너↑)");
                        ui.add(
                            egui::Slider::new(&mut self.vec_cfg.corner_threshold, 0..=180)
                                .suffix("°"),
                        );
                        ui.add_space(4.0);
                        field_label(ui, "최소 선분 길이");
                        ui.add(
                            egui::Slider::new(&mut self.vec_cfg.length_threshold, 3.5..=10.0)
                                .fixed_decimals(1),
                        );
                        ui.add_space(4.0);
                        field_label(ui, "이음 임계값");
                        ui.add(
                            egui::Slider::new(&mut self.vec_cfg.splice_threshold, 0..=180)
                                .suffix("°"),
                        );
                        if self.vec_cfg.color_mode == VecColorMode::Color {
                            ui.add_space(4.0);
                            field_label(ui, "색 레이어 간격");
                            ui.add(egui::Slider::new(&mut self.vec_cfg.layer_difference, 0..=128));
                        }
                    });

                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(
                        "💡 벡터 변환은 로고·아이콘·라인아트 같은 단색 그래픽에 가장 잘 맞아요. \
                         사진은 path가 매우 많아져 결과 파일이 커질 수 있습니다.",
                    )
                    .size(11.0)
                    .color(theme::TEXT_MUTED),
                );
            });
        });
    }

    /// The "목표 크기" controls: factor vs explicit dimensions, with a live readout.
    fn size_controls(&mut self, ui: &mut egui::Ui) {
        // Segmented mode toggle.
        ui.horizontal(|ui| {
            let w = (ui.available_width() - ui.spacing().item_spacing.x) / 2.0;
            if ui
                .add_sized(
                    [w, 26.0],
                    egui::SelectableLabel::new(self.size_mode == SizeMode::Scale, "배율로"),
                )
                .clicked()
            {
                self.size_mode = SizeMode::Scale;
            }
            if ui
                .add_sized(
                    [w, 26.0],
                    egui::SelectableLabel::new(
                        self.size_mode == SizeMode::Dimensions,
                        "크기 직접 지정",
                    ),
                )
                .clicked()
            {
                self.size_mode = SizeMode::Dimensions;
                self.sync_dimensions_to_aspect(true);
            }
        });

        ui.add_space(8.0);

        match self.size_mode {
            SizeMode::Scale => {
                ui.horizontal(|ui| {
                    for s in [2.0f32, 3.0, 4.0] {
                        let sel = (self.target_scale - s).abs() < 0.05;
                        if ui.selectable_label(sel, format!("{}×", s as u32)).clicked() {
                            self.target_scale = s;
                        }
                    }
                });
                ui.add_space(2.0);
                ui.add(
                    egui::Slider::new(&mut self.target_scale, 1.0..=8.0)
                        .step_by(0.1)
                        .fixed_decimals(1)
                        .suffix("×"),
                );
            }
            SizeMode::Dimensions => {
                let (rw, rh) = ui
                    .horizontal(|ui| {
                        ui.label("가로");
                        let rw = ui.add(
                            egui::DragValue::new(&mut self.target_w)
                                .range(1..=100_000)
                                .speed(2.0)
                                .suffix(" px"),
                        );
                        ui.label("×");
                        ui.label("세로");
                        let rh = ui.add(
                            egui::DragValue::new(&mut self.target_h)
                                .range(1..=100_000)
                                .speed(2.0)
                                .suffix(" px"),
                        );
                        (rw, rh)
                    })
                    .inner;

                let lock = ui.checkbox(&mut self.lock_aspect, "비율 유지 (한 쪽 입력 시 자동 계산)");

                if self.lock_aspect {
                    if let Some((sw, sh)) = self.src_dims {
                        let aspect = sw as f32 / sh.max(1) as f32;
                        if rw.changed() {
                            self.target_h = ((self.target_w as f32 / aspect).round() as u32).max(1);
                        } else if rh.changed() || lock.changed() {
                            self.target_w = ((self.target_h as f32 * aspect).round() as u32).max(1);
                        }
                    }
                }
            }
        }

        ui.add_space(8.0);
        self.size_readout(ui);
    }

    /// Live "원본 → 결과 (×배율)" line.
    fn size_readout(&self, ui: &mut egui::Ui) {
        match self.src_dims {
            Some((sw, sh)) => {
                let target = self.current_target();
                let (tw, th) = target.resolve(sw, sh);
                let factor = target.factor_for(sw, sh);
                egui::Frame::none()
                    .fill(theme::BG_WINDOW)
                    .rounding(Rounding::same(6.0))
                    .inner_margin(Margin::symmetric(8.0, 5.0))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing.x = 5.0;
                            ui.label(
                                egui::RichText::new(format!("{sw}×{sh}"))
                                    .monospace()
                                    .color(theme::TEXT_MUTED),
                            );
                            ui.label(egui::RichText::new("→").color(theme::ACCENT).strong());
                            ui.label(
                                egui::RichText::new(format!("{tw}×{th}"))
                                    .monospace()
                                    .strong()
                                    .color(theme::TEXT),
                            );
                            ui.label(
                                egui::RichText::new(format!("(×{factor:.1})"))
                                    .color(theme::TEXT_MUTED),
                            );
                        });
                    });
            }
            None => {
                ui.label(
                    egui::RichText::new("이미지를 선택하면 결과 크기가 여기에 표시돼요.")
                        .size(12.0)
                        .color(theme::TEXT_MUTED),
                );
            }
        }
    }

    /// Auto-download collapsing section.
    fn download_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::CollapsingHeader::new("모델 자동 다운로드")
            .default_open(self.model_path.is_none())
            .show(ui, |ui| {
                let specs = download::MODELS;
                field_label(ui, "받을 모델");
                let w = (ui.available_width() - 6.0).max(80.0);
                egui::ComboBox::from_id_source("dl_model")
                    .selected_text(specs[self.selected_model_idx].label)
                    .width(w)
                    .truncate()
                    .show_ui(ui, |ui| {
                        for (i, s) in specs.iter().enumerate() {
                            ui.selectable_value(&mut self.selected_model_idx, i, s.label);
                        }
                    });
                let spec = specs[self.selected_model_idx];
                ui.add_space(4.0);
                if download::is_downloaded(&spec) {
                    if ui
                        .add(wide_button(ui, "✔  받아둔 모델 사용"))
                        .clicked()
                    {
                        self.model_path = Some(download::model_path(&spec));
                        self.model_scale = spec.scale;
                        self.status = format!("모델 준비 완료: {}", spec.label);
                    }
                } else if ui
                    .add(accent_button(ui, &format!("⬇  다운로드 (~{} MB)", spec.approx_mb), true))
                    .clicked()
                {
                    self.start_download(ctx);
                }
            });
    }

    /// The pinned primary action (+ cancel while running).
    fn action_row(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if self.running {
            ui.horizontal(|ui| {
                let full = ui.available_width();
                ui.add_enabled_ui(false, |ui| {
                    ui.add(accent_button(ui, "처리 중…", false).min_size(Vec2::new(full * 0.6, 34.0)));
                });
                if ui
                    .add(
                        egui::Button::new(egui::RichText::new("■ 취소").color(theme::TEXT))
                            .fill(theme::CARD_HOVER)
                            .rounding(Rounding::same(theme::ROUNDING))
                            .min_size(Vec2::new(ui.available_width(), 34.0)),
                    )
                    .clicked()
                {
                    self.cancel_flag.store(true, Ordering::SeqCst);
                    self.status = "취소하는 중…".to_string();
                }
            });
        } else {
            let needs_model = self.mode == AppMode::Upscale;
            let ready = !self.input_files.is_empty()
                && self.output_dir.is_some()
                && (!needs_model || self.model_path.is_some());
            let label = match self.mode {
                AppMode::Upscale => "▶  업스케일 시작",
                AppMode::Vectorize => "▶  벡터로 변환",
            };
            let resp = ui.add(
                accent_button(ui, label, ready)
                    .min_size(Vec2::new(ui.available_width(), 34.0)),
            );
            if resp.clicked() {
                self.start(ctx);
            }
            if !ready {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(self.missing_hint())
                        .size(11.5)
                        .color(theme::WARN),
                );
            }
        }
    }

    fn missing_hint(&self) -> String {
        let mut need: Vec<&str> = Vec::new();
        if self.input_files.is_empty() {
            need.push("이미지");
        }
        if self.output_dir.is_none() {
            need.push("출력 폴더");
        }
        if self.mode == AppMode::Upscale && self.model_path.is_none() {
            need.push("모델");
        }
        format!("시작하려면 {} 을(를) 지정하세요.", need.join(" · "))
    }

    // ---- Preview (signature: before/after compare) --------------------------

    fn preview_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(theme::title("미리보기", 14.0));
            if self.input_preview.is_some() && self.output_preview.is_some() {
                ui.label(
                    egui::RichText::new("· 핸들을 드래그해 원본과 결과를 비교하세요")
                        .size(11.0)
                        .color(theme::TEXT_MUTED),
                );
            }
        });
        ui.add_space(8.0);

        let before = self.input_preview.clone();
        let after = self.output_preview.clone();

        match (before, after) {
            (Some(before), Some(after)) => {
                let mut split = self.compare_split;
                comparison(ui, &before, &after, &mut split);
                self.compare_split = split;
            }
            (Some(before), None) => {
                let caption = match self.mode {
                    AppMode::Upscale => "원본 · 업스케일하면 비교가 표시돼요",
                    AppMode::Vectorize => "원본 · 벡터로 변환하면 비교가 표시돼요",
                };
                single_image(ui, &before, caption)
            }
            _ => empty_state(ui),
        }
    }

    // ---- Worker plumbing ----------------------------------------------------

    /// Read the first selected image's dimensions and pre-fill target size.
    fn refresh_source_dims(&mut self) {
        self.src_dims = self
            .input_files
            .first()
            .and_then(|p| image::image_dimensions(p).ok());
        if let Some((w, h)) = self.src_dims {
            self.target_w = (w * self.model_scale).max(1);
            self.target_h = (h * self.model_scale).max(1);
        }
    }

    /// If aspect is locked, recompute height from width using the source ratio.
    fn sync_dimensions_to_aspect(&mut self, _force: bool) {
        if self.lock_aspect {
            if let Some((sw, sh)) = self.src_dims {
                let aspect = sw as f32 / sh.max(1) as f32;
                self.target_h = ((self.target_w as f32 / aspect).round() as u32).max(1);
            }
        }
    }

    /// The target size implied by the current controls.
    fn current_target(&self) -> TargetSize {
        match self.size_mode {
            SizeMode::Scale => TargetSize::Scale(self.target_scale),
            SizeMode::Dimensions => {
                if self.lock_aspect {
                    TargetSize::FitBox { w: self.target_w, h: self.target_h }
                } else {
                    TargetSize::Exact { w: self.target_w, h: self.target_h }
                }
            }
        }
    }

    /// Dispatch the primary action for the active workflow.
    fn start(&mut self, ctx: &egui::Context) {
        match self.mode {
            AppMode::Upscale => self.start_upscale(ctx),
            AppMode::Vectorize => self.start_vectorize(ctx),
        }
    }

    /// Validate inputs and kick off the vectorize worker (raster → SVG).
    fn start_vectorize(&mut self, ctx: &egui::Context) {
        let Some(out_dir) = self.output_dir.clone() else {
            self.status = "⚠ 출력 폴더를 선택하세요.".to_string();
            return;
        };
        if self.input_files.is_empty() {
            self.status = "⚠ 변환할 이미지를 선택하세요.".to_string();
            return;
        }

        let files = self.input_files.clone();
        let cfg = self.vec_cfg;
        let suffix = self.suffix.clone();

        self.cancel_flag = Arc::new(AtomicBool::new(false));
        let cancel = self.cancel_flag.clone();
        let (tx, rx) = channel();
        self.rx = Some(rx);
        self.running = true;
        self.progress = 0.0;
        self.output_preview = None;
        self.status = "벡터로 변환하는 중…".to_string();

        let ctx = ctx.clone();
        thread::spawn(move || {
            let total = files.len() as f32;
            for (i, input) in files.iter().enumerate() {
                if cancel.load(Ordering::SeqCst) {
                    let _ = tx.send(WorkerMsg::Log("취소되었습니다.".to_string()));
                    break;
                }

                let output = svg_output_path(input, &out_dir, &suffix);
                let name = input
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                let _ = tx.send(WorkerMsg::Log(format!(
                    "변환 중 ({}/{}): {name}",
                    i + 1,
                    files.len()
                )));

                match vectorize::vectorize_file(input, &output, &cfg) {
                    Ok(()) => {
                        let _ = tx.send(WorkerMsg::FileDone {
                            output: output.clone(),
                        });
                    }
                    Err(e) => {
                        let _ = tx.send(WorkerMsg::Log(format!("실패: {name}: {e:#}")));
                    }
                }
                let _ = tx.send(WorkerMsg::Progress((i as f32 + 1.0) / total));
                ctx.request_repaint();
            }

            let _ = tx.send(WorkerMsg::Finished);
            ctx.request_repaint();
        });
    }

    /// Validate inputs and kick off the upscale worker.
    fn start_upscale(&mut self, ctx: &egui::Context) {
        let Some(out_dir) = self.output_dir.clone() else {
            self.status = "⚠ 출력 폴더를 선택하세요.".to_string();
            return;
        };
        let Some(model_path) = self.model_path.clone() else {
            self.status = "⚠ ONNX 모델 파일을 선택하세요.".to_string();
            return;
        };
        if self.input_files.is_empty() {
            self.status = "⚠ 업스케일할 이미지를 선택하세요.".to_string();
            return;
        }

        let files = self.input_files.clone();
        let format = self.output_format;
        let keep_input = self.keep_input_format;
        let suffix = self.suffix.clone();
        let quality = self.quality;
        let scale = self.model_scale;
        let tile_size = self.tile_size;
        let use_gpu = self.use_gpu;
        let target = self.current_target();

        self.cancel_flag = Arc::new(AtomicBool::new(false));
        let cancel = self.cancel_flag.clone();
        let (tx, rx) = channel();
        self.rx = Some(rx);
        self.running = true;
        self.progress = 0.0;
        self.output_preview = None;
        self.status = "모델을 불러오는 중…".to_string();

        let ctx = ctx.clone();
        thread::spawn(move || {
            let upscaler = match Engine::load(&model_path, tile_size, scale, use_gpu) {
                Ok(u) => {
                    let _ = tx.send(WorkerMsg::Log(format!("백엔드: {}", u.backend_name())));
                    u
                }
                Err(e) => {
                    let _ = tx.send(WorkerMsg::Log(format!("모델 로드 실패: {e:#}")));
                    let _ = tx.send(WorkerMsg::Finished);
                    ctx.request_repaint();
                    return;
                }
            };

            let total = files.len() as f32;
            for (i, input) in files.iter().enumerate() {
                if cancel.load(Ordering::SeqCst) {
                    let _ = tx.send(WorkerMsg::Log("취소되었습니다.".to_string()));
                    break;
                }

                let (output, fmt) = resolve_output(input, &out_dir, format, keep_input, &suffix);
                let name = input
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                let _ = tx.send(WorkerMsg::Log(format!(
                    "처리 중 ({}/{}): {name}",
                    i + 1,
                    files.len()
                )));

                let base = i as f32;
                let tx_p = tx.clone();
                let ctx_p = ctx.clone();
                let result = pipeline::process(&upscaler, input, &output, target, fmt, quality, |p| {
                    let _ = tx_p.send(WorkerMsg::Progress((base + p) / total));
                    ctx_p.request_repaint();
                });

                match result {
                    Ok(()) => {
                        let _ = tx.send(WorkerMsg::FileDone { output: output.clone() });
                    }
                    Err(e) => {
                        let _ = tx.send(WorkerMsg::Log(format!("실패: {name}: {e:#}")));
                    }
                }
                let _ = tx.send(WorkerMsg::Progress((base + 1.0) / total));
                ctx.request_repaint();
            }

            let _ = tx.send(WorkerMsg::Finished);
            ctx.request_repaint();
        });
    }

    /// Download the selected curated model on a background thread.
    fn start_download(&mut self, ctx: &egui::Context) {
        let spec = download::MODELS[self.selected_model_idx];

        self.cancel_flag = Arc::new(AtomicBool::new(false));
        let cancel = self.cancel_flag.clone();
        let (tx, rx) = channel();
        self.dl_rx = Some(rx);
        self.running = true;
        self.progress = 0.0;
        self.status = format!("{} 다운로드 시작…", spec.label);

        let ctx = ctx.clone();
        thread::spawn(move || {
            let result = download::ensure_model(&spec, &cancel, |downloaded, total| {
                let _ = tx.send(DlMsg::Progress { downloaded, total });
                ctx.request_repaint();
            });
            match result {
                Ok(path) => {
                    let _ = tx.send(DlMsg::Done { path, scale: spec.scale });
                }
                Err(e) => {
                    let _ = tx.send(DlMsg::Failed(format!("{e:#}")));
                }
            }
            ctx.request_repaint();
        });
    }

    /// Pull any pending worker messages and update UI state.
    fn drain_worker_messages(&mut self, ctx: &egui::Context) {
        let mut finished = false;
        let mut newest_output: Option<PathBuf> = None;

        if let Some(rx) = &self.rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    WorkerMsg::Progress(p) => self.progress = p.clamp(0.0, 1.0),
                    WorkerMsg::Log(s) => self.status = s,
                    WorkerMsg::FileDone { output } => newest_output = Some(output),
                    WorkerMsg::Finished => finished = true,
                }
            }
        }

        if let Some(output) = newest_output {
            self.last_output = Some(output.clone());
            self.output_preview = load_preview_texture(ctx, &output, "output_preview");
            self.compare_split = 0.5;
        }

        if finished {
            self.running = false;
            self.rx = None;
            self.progress = if self.cancel_flag.load(Ordering::SeqCst) {
                self.progress
            } else {
                1.0
            };
            if !self.status.starts_with('취') && !self.status.starts_with("모델 로드 실패") {
                self.status = "완료되었어요. 미리보기에서 결과를 확인하세요.".to_string();
            }
        }

        // Drain the model-download worker, if one is active.
        let mut dl_progress: Option<(u64, Option<u64>)> = None;
        let mut dl_done: Option<(PathBuf, u32)> = None;
        let mut dl_failed: Option<String> = None;
        if let Some(rx) = &self.dl_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    DlMsg::Progress { downloaded, total } => dl_progress = Some((downloaded, total)),
                    DlMsg::Done { path, scale } => dl_done = Some((path, scale)),
                    DlMsg::Failed(e) => dl_failed = Some(e),
                }
            }
        }
        if let Some((d, t)) = dl_progress {
            self.progress = match t {
                Some(t) if t > 0 => (d as f32 / t as f32).clamp(0.0, 1.0),
                _ => self.progress,
            };
            self.status = match t {
                Some(t) => format!(
                    "다운로드 중… {} / {}",
                    download::human_bytes(d),
                    download::human_bytes(t)
                ),
                None => format!("다운로드 중… {}", download::human_bytes(d)),
            };
        }
        if let Some((path, scale)) = dl_done {
            self.model_path = Some(path);
            self.model_scale = scale;
            self.running = false;
            self.dl_rx = None;
            self.progress = 1.0;
            self.status = "모델 다운로드 완료. 사용할 준비가 됐어요.".to_string();
        } else if let Some(e) = dl_failed {
            self.running = false;
            self.dl_rx = None;
            self.status = format!("다운로드 실패: {e}");
        }
    }

    /// Load a thumbnail of the first selected file for the "before" pane.
    fn ensure_input_preview(&mut self, ctx: &egui::Context) {
        let first = self.input_files.first().cloned();
        if first != self.preview_source {
            self.preview_source = first.clone();
            self.input_preview = first.and_then(|p| load_texture(ctx, &p, "input_preview"));
        }
    }
}

// ---- Reusable widgets -------------------------------------------------------

/// A titled section card with a numbered step chip.
fn card<R>(ui: &mut egui::Ui, step: &str, title: &str, body: impl FnOnce(&mut egui::Ui) -> R) {
    egui::Frame::none()
        .fill(theme::CARD)
        .rounding(Rounding::same(theme::ROUNDING))
        .stroke(Stroke::new(1.0, theme::HAIRLINE))
        .inner_margin(Margin::same(10.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                egui::Frame::none()
                    .fill(theme::ACCENT.gamma_multiply(0.18))
                    .rounding(Rounding::same(5.0))
                    .inner_margin(Margin::symmetric(6.0, 1.5))
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(step)
                                .size(10.5)
                                .strong()
                                .color(theme::ACCENT),
                        );
                    });
                ui.add_space(2.0);
                ui.label(theme::title(title, 13.0));
            });
            ui.add_space(7.0);
            body(ui);
        });
    ui.add_space(8.0);
}

/// A small muted field label, placed above its control.
fn field_label(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).size(11.0).color(theme::TEXT_MUTED));
}

/// A full-width neutral button.
fn wide_button(ui: &egui::Ui, text: &str) -> egui::Button<'static> {
    egui::Button::new(egui::RichText::new(text.to_owned()).color(theme::TEXT))
        .fill(theme::CARD_HOVER)
        .stroke(Stroke::new(1.0, theme::HAIRLINE))
        .rounding(Rounding::same(theme::ROUNDING))
        .min_size(Vec2::new(ui.available_width(), 28.0))
}

/// A full-width accent (primary) button; dimmed when `enabled` is false.
fn accent_button(ui: &egui::Ui, text: &str, enabled: bool) -> egui::Button<'static> {
    let (fill, fg) = if enabled {
        (theme::ACCENT, theme::ON_ACCENT)
    } else {
        (theme::CARD_HOVER, theme::TEXT_MUTED)
    };
    egui::Button::new(egui::RichText::new(text.to_owned()).color(fg).strong())
        .fill(fill)
        .rounding(Rounding::same(theme::ROUNDING))
        .min_size(Vec2::new(ui.available_width(), 30.0))
}

// ---- Preview helpers --------------------------------------------------------

/// Draggable before/after comparison: `after` fills the rect; `before` is shown
/// clipped to the left of a movable split. Both map to the same rect so the same
/// content aligns and the sharpness difference is visible across the divider.
fn comparison(
    ui: &mut egui::Ui,
    before: &egui::TextureHandle,
    after: &egui::TextureHandle,
    split: &mut f32,
) {
    let size = fit_size(ui, after.size_vec2());
    ui.vertical_centered(|ui| {
        let (rect, resp) = ui.allocate_exact_size(size, Sense::click_and_drag());

        if let Some(pos) = resp.interact_pointer_pos() {
            if resp.dragged() || resp.clicked() {
                *split = ((pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
            }
        }

        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        let painter = ui.painter();

        // Rounded backdrop + the "after" image.
        painter.rect_filled(rect, Rounding::same(10.0), theme::BG_PANEL);
        painter.image(after.id(), rect, uv, Color32::WHITE);

        // "before" clipped to the left of the split.
        let split_x = rect.left() + rect.width() * *split;
        let left = Rect::from_min_max(rect.min, Pos2::new(split_x, rect.max.y));
        painter.with_clip_rect(left).image(before.id(), rect, uv, Color32::WHITE);

        // Divider + grab handle.
        painter.line_segment(
            [Pos2::new(split_x, rect.top()), Pos2::new(split_x, rect.bottom())],
            Stroke::new(2.0, theme::ACCENT),
        );
        painter.circle_filled(Pos2::new(split_x, rect.center().y), 7.0, theme::ACCENT);
        painter.circle_filled(Pos2::new(split_x, rect.center().y), 3.0, theme::ON_ACCENT);

        // Corner labels.
        corner_tag(painter, rect, Align2::LEFT_TOP, "원본");
        corner_tag(painter, rect, Align2::RIGHT_TOP, "결과");

        painter.rect_stroke(rect, Rounding::same(10.0), Stroke::new(1.0, theme::HAIRLINE));
    });
}

/// Show a single image centered, with a caption beneath.
fn single_image(ui: &mut egui::Ui, tex: &egui::TextureHandle, caption: &str) {
    let size = fit_size(ui, tex.size_vec2());
    ui.vertical_centered(|ui| {
        let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        ui.painter().rect_filled(rect, Rounding::same(10.0), theme::BG_PANEL);
        ui.painter().image(tex.id(), rect, uv, Color32::WHITE);
        ui.painter()
            .rect_stroke(rect, Rounding::same(10.0), Stroke::new(1.0, theme::HAIRLINE));
        corner_tag(ui.painter(), rect, Align2::LEFT_TOP, "원본");
        ui.add_space(8.0);
        ui.label(egui::RichText::new(caption).size(12.0).color(theme::TEXT_MUTED));
    });
}

/// Friendly empty state when nothing is loaded yet.
fn empty_state(ui: &mut egui::Ui) {
    ui.vertical_centered(|ui| {
        ui.add_space(ui.available_height() * 0.3);
        ui.label(egui::RichText::new("🖼").size(34.0).color(theme::HAIRLINE));
        ui.add_space(6.0);
        ui.label(theme::title("아직 미리볼 이미지가 없어요", 14.0));
        ui.add_space(3.0);
        ui.label(
            egui::RichText::new("왼쪽에서 이미지를 선택하면 여기에서 원본을 보고,\n업스케일 후 결과와 나란히 비교할 수 있어요.")
                .size(11.5)
                .color(theme::TEXT_MUTED),
        );
    });
}

/// A small pill label drawn in a corner of `rect`.
fn corner_tag(painter: &egui::Painter, rect: Rect, align: Align2, text: &str) {
    let pad = 8.0;
    let anchor = match align {
        Align2::RIGHT_TOP => Pos2::new(rect.right() - pad, rect.top() + pad),
        _ => Pos2::new(rect.left() + pad, rect.top() + pad),
    };
    let galley = painter.layout_no_wrap(
        text.to_owned(),
        FontId::proportional(11.0),
        theme::TEXT,
    );
    let size = galley.size();
    let min = match align {
        Align2::RIGHT_TOP => Pos2::new(anchor.x - size.x - 8.0, anchor.y),
        _ => anchor,
    };
    let bg = Rect::from_min_size(min, size + Vec2::new(12.0, 6.0));
    painter.rect_filled(bg, Rounding::same(6.0), theme::BG_WINDOW.gamma_multiply(0.85));
    painter.galley(min + Vec2::new(6.0, 3.0), galley, theme::TEXT);
}

/// Fit `tex_size` within the available area, preserving aspect ratio.
fn fit_size(ui: &egui::Ui, tex_size: Vec2) -> Vec2 {
    let avail = ui.available_size();
    let max_w = avail.x.max(64.0);
    let max_h = (avail.y - 30.0).max(64.0);
    let aspect = if tex_size.y > 0.0 { tex_size.x / tex_size.y } else { 1.0 };
    let mut w = max_w;
    let mut h = w / aspect;
    if h > max_h {
        h = max_h;
        w = h * aspect;
    }
    Vec2::new(w, h)
}

/// Load an image file into a downscaled egui texture for preview, or `None`.
fn load_texture(ctx: &egui::Context, path: &Path, name: &str) -> Option<egui::TextureHandle> {
    let img = image::open(path).ok()?;
    let thumb = img.thumbnail(720, 720).to_rgba8();
    let size = [thumb.width() as usize, thumb.height() as usize];
    let color = egui::ColorImage::from_rgba_unmultiplied(size, thumb.as_raw());
    Some(ctx.load_texture(name, color, egui::TextureOptions::LINEAR))
}

/// Load an output file into a preview texture: raster files via `image`, SVG
/// files via the pure-Rust vector renderer.
fn load_preview_texture(
    ctx: &egui::Context,
    path: &Path,
    name: &str,
) -> Option<egui::TextureHandle> {
    let is_svg = path
        .extension()
        .map(|e| e.eq_ignore_ascii_case("svg"))
        .unwrap_or(false);
    if !is_svg {
        return load_texture(ctx, path, name);
    }
    let svg = std::fs::read_to_string(path).ok()?;
    let (rgba, w, h) = vectorize::render_svg_to_rgba(&svg, 720).ok()?;
    let color = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
    Some(ctx.load_texture(name, color, egui::TextureOptions::LINEAR))
}

/// Output path for a traced image: `<stem><suffix>.svg` inside `out_dir`.
fn svg_output_path(input: &Path, out_dir: &Path, suffix: &str) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".to_string());
    out_dir.join(format!("{stem}{suffix}.svg"))
}

/// Short display string for an optional output directory.
fn path_hint(dir: &Option<PathBuf>) -> String {
    match dir {
        Some(p) => {
            let s = p.display().to_string();
            if s.len() > 38 {
                format!("…{}", &s[s.len() - 37..])
            } else {
                s
            }
        }
        None => "(미선택)".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lay out the entire UI at the given window size, optionally with the
    /// before/after preview populated. Panics if any panel/card/widget fails to
    /// lay out.
    ///
    /// Note: this guards against layout *panics*, not text clipping — egui clips
    /// overflowing text gracefully (no error), so the side panel is kept clear by
    /// construction instead (every combo/slider has its label stacked above and a
    /// width bounded to the panel; see `controls_ui`).
    fn lay_out(mode: AppMode, with_previews: bool, w: f32, h: f32) {
        let ctx = egui::Context::default();
        crate::fonts::install_korean_fonts(&ctx);
        crate::theme::apply(&ctx);

        let mut app = UpscalerApp::default();
        app.mode = mode;
        if with_previews {
            let img = egui::ColorImage::new([8, 6], Color32::GRAY);
            app.input_preview =
                Some(ctx.load_texture("t_in", img.clone(), egui::TextureOptions::LINEAR));
            app.output_preview =
                Some(ctx.load_texture("t_out", img, egui::TextureOptions::LINEAR));
            app.src_dims = Some((1920, 1080));
            app.size_mode = SizeMode::Dimensions;
        }

        let mut input = egui::RawInput::default();
        input.screen_rect = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(w, h)));
        // Two frames: the first builds the font atlas, the second lays out
        // against real sizes.
        let _ = ctx.run(input.clone(), |ctx| app.ui(ctx));
        let _ = ctx.run(input, |ctx| app.ui(ctx));
    }

    #[test]
    fn default_preselects_cached_model() {
        let spec = crate::upscaler::download::MODELS[0];
        let app = UpscalerApp::default();
        // The model is pre-selected exactly when it is present in the cache.
        assert_eq!(
            app.model_path.is_some(),
            crate::upscaler::download::is_downloaded(&spec)
        );
        if app.model_path.is_some() {
            assert_eq!(app.model_scale, spec.scale);
        }
    }

    #[test]
    fn ui_lays_out_across_window_sizes() {
        use AppMode::*;
        lay_out(Upscale, false, 560.0, 420.0); // minimum window
        lay_out(Upscale, false, 340.0, 400.0); // tiny: side panel clamps to its minimum width
        lay_out(Upscale, false, 820.0, 560.0); // default window
        lay_out(Upscale, true, 560.0, 420.0); // with the before/after compare populated

        // The vectorize workflow must lay out cleanly too, including its compare.
        lay_out(Vectorize, false, 560.0, 420.0);
        lay_out(Vectorize, false, 820.0, 560.0);
        lay_out(Vectorize, true, 560.0, 420.0);
    }
}
