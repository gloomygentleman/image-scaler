//! Visual theme for the app.
//!
//! Design direction — "초해상도 작업대" (a super-resolution workbench): a calm,
//! deep blue-charcoal imaging tool where a single luminous violet accent marks
//! everything the user acts on (primary action, active toggles, the before/after
//! handle, progress). Success states use teal and validation uses amber, so
//! colour always carries meaning rather than decoration. Detail emerging from
//! pixels is the subject; the signature is the draggable before/after compare.

use eframe::egui::{self, Color32, FontFamily, FontId, Rounding, Stroke, TextStyle, Vec2};

use crate::fonts::HEADING_FAMILY;

// ---- Palette ("Darkroom Indigo") -------------------------------------------

/// App background — the darkroom.
pub const BG_WINDOW: Color32 = Color32::from_rgb(0x0F, 0x12, 0x16);
/// Panels behind the cards.
pub const BG_PANEL: Color32 = Color32::from_rgb(0x14, 0x18, 0x1F);
/// Raised section cards.
pub const CARD: Color32 = Color32::from_rgb(0x1B, 0x21, 0x2B);
/// Hovered / elevated surface.
pub const CARD_HOVER: Color32 = Color32::from_rgb(0x25, 0x2E, 0x3B);
/// Hairline dividers and card borders.
pub const HAIRLINE: Color32 = Color32::from_rgb(0x2C, 0x36, 0x44);
/// Primary text.
pub const TEXT: Color32 = Color32::from_rgb(0xEA, 0xEE, 0xF5);
/// Secondary / muted text.
pub const TEXT_MUTED: Color32 = Color32::from_rgb(0x96, 0xA1, 0xB2);

/// Signature accent — luminous indigo-violet ("enhance").
pub const ACCENT: Color32 = Color32::from_rgb(0x8C, 0x7C, 0xFF);
pub const ACCENT_HOVER: Color32 = Color32::from_rgb(0xA2, 0x95, 0xFF);
pub const ACCENT_PRESS: Color32 = Color32::from_rgb(0x76, 0x66, 0xE6);
/// Readable text placed on top of the accent fill.
pub const ON_ACCENT: Color32 = Color32::from_rgb(0x0F, 0x12, 0x16);

/// Success (download complete, finished).
pub const GOOD: Color32 = Color32::from_rgb(0x46, 0xD9, 0xA8);
/// Validation / attention.
pub const WARN: Color32 = Color32::from_rgb(0xF2, 0xB8, 0x5A);

/// Rounding shared by cards and buttons.
pub const ROUNDING: f32 = 8.0;

/// Apply the theme (colours, spacing, type scale) to the egui context.
pub fn apply(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    // ---- Type scale: bold headings, compact body, monospace for data ----
    let heading = FontFamily::Name(HEADING_FAMILY.into());
    style.text_styles = [
        (TextStyle::Heading, FontId::new(17.0, heading.clone())),
        (TextStyle::Body, FontId::new(12.5, FontFamily::Proportional)),
        (TextStyle::Button, FontId::new(12.5, FontFamily::Proportional)),
        (TextStyle::Small, FontId::new(10.5, FontFamily::Proportional)),
        (TextStyle::Monospace, FontId::new(11.5, FontFamily::Monospace)),
    ]
    .into();

    // ---- Spacing: tight but legible, so groups still read as distinct steps --
    style.spacing.item_spacing = Vec2::new(6.0, 5.0);
    style.spacing.button_padding = Vec2::new(9.0, 4.0);
    style.spacing.interact_size.y = 24.0;
    style.spacing.slider_width = 135.0;
    style.spacing.combo_width = 165.0;
    style.spacing.indent = 13.0;

    // ---- Colours ----
    let mut v = egui::Visuals::dark();
    v.dark_mode = true;
    v.panel_fill = BG_PANEL;
    v.window_fill = CARD;
    v.window_stroke = Stroke::new(1.0, HAIRLINE);
    v.extreme_bg_color = BG_WINDOW; // text edits, progress bar trough
    v.faint_bg_color = Color32::from_rgb(0x18, 0x1E, 0x27);
    v.hyperlink_color = ACCENT;
    v.selection.bg_fill = ACCENT.gamma_multiply(0.35);
    v.selection.stroke = Stroke::new(1.0, ACCENT);

    let rounding = Rounding::same(ROUNDING);

    // Non-interactive (labels, separators, card frames).
    v.widgets.noninteractive.bg_fill = CARD;
    v.widgets.noninteractive.weak_bg_fill = CARD;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, HAIRLINE);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.noninteractive.rounding = rounding;

    // Inactive (resting buttons, combos).
    v.widgets.inactive.bg_fill = CARD_HOVER;
    v.widgets.inactive.weak_bg_fill = CARD_HOVER;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, HAIRLINE);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.inactive.rounding = rounding;

    // Hovered.
    v.widgets.hovered.bg_fill = Color32::from_rgb(0x2E, 0x38, 0x47);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(0x2E, 0x38, 0x47);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT_HOVER.gamma_multiply(0.8));
    v.widgets.hovered.fg_stroke = Stroke::new(1.5, TEXT);
    v.widgets.hovered.rounding = rounding;

    // Active / pressed.
    v.widgets.active.bg_fill = ACCENT_PRESS;
    v.widgets.active.weak_bg_fill = ACCENT_PRESS;
    v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    v.widgets.active.fg_stroke = Stroke::new(1.5, ON_ACCENT);
    v.widgets.active.rounding = rounding;

    // Open (expanded combo/menu).
    v.widgets.open.bg_fill = CARD_HOVER;
    v.widgets.open.weak_bg_fill = CARD_HOVER;
    v.widgets.open.bg_stroke = Stroke::new(1.0, HAIRLINE);
    v.widgets.open.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.open.rounding = rounding;

    v.window_rounding = Rounding::same(12.0);
    v.menu_rounding = rounding;

    style.visuals = v;
    ctx.set_style(style);
}

/// A `RichText` styled as a card/section title (bold heading family, given size).
pub fn title(text: impl Into<String>, size: f32) -> egui::RichText {
    egui::RichText::new(text.into())
        .font(FontId::new(size, FontFamily::Name(HEADING_FAMILY.into())))
        .color(TEXT)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GUI start-up path (embed fonts → apply theme → lay out a frame that
    /// uses the bold heading family) must render without panicking. This guards
    /// the font/theme ordering and the `"heading"` family wiring headlessly,
    /// since the windowed GUI can't be launched in CI.
    #[test]
    fn theme_and_heading_render_without_panic() {
        let ctx = egui::Context::default();
        crate::fonts::install_korean_fonts(&ctx);
        apply(&ctx);
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.label(title("목표 크기", 15.0));
                ui.heading("이미지 업스케일러");
                ui.label("원본 64×64 → 결과 256×256");
            });
        });
    }
}
