//! Embeds Korean-capable fonts into egui so Hangul renders correctly.
//!
//! egui's bundled fonts only cover Latin glyphs, so without this every Korean
//! label shows as missing-glyph boxes (□). We compile Hangul fonts directly into
//! the executable (`include_bytes!`) and register them: a regular weight for
//! body text and a bold weight exposed as a named `"heading"` family for titles.
//! egui's default fonts stay on as fallbacks (emoji / icons).
//!
//! Bundling fonts at build time keeps the single self-contained binary
//! guarantee: Korean renders on any machine regardless of which fonts the OS has
//! installed — nothing to download or install at run time.
//!
//! Fonts: Nanum Gothic (나눔고딕) by NAVER, SIL Open Font License 1.1 — see
//! `assets/fonts/OFL.txt`. The OFL permits embedding and redistribution as part
//! of an application.

use eframe::egui;

/// Body / regular weight, compiled into the binary.
const KOREAN_REGULAR: &[u8] = include_bytes!("../assets/fonts/NanumGothic-Regular.ttf");
/// Bold weight, used for headings and titles.
const KOREAN_BOLD: &[u8] = include_bytes!("../assets/fonts/NanumGothic-Bold.ttf");

/// Name of the custom font family used for headings (bold).
pub const HEADING_FAMILY: &str = "heading";

const REGULAR_KEY: &str = "nanum_regular";
const BOLD_KEY: &str = "nanum_bold";

/// Register the embedded Korean fonts with `ctx`.
///
/// This always succeeds: the fonts travel inside the executable, so there is no
/// I/O and no dependency on system-installed fonts. After this, body text uses
/// the regular weight and `FontFamily::Name("heading")` uses the bold weight.
pub fn install_korean_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    fonts
        .font_data
        .insert(REGULAR_KEY.to_owned(), egui::FontData::from_static(KOREAN_REGULAR));
    fonts
        .font_data
        .insert(BOLD_KEY.to_owned(), egui::FontData::from_static(KOREAN_BOLD));

    // Body: Korean regular first, egui's bundled fonts after as fallbacks for
    // the symbols/emoji the UI uses (▶ ■ 🖼 📁 …).
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, REGULAR_KEY.to_owned());
    }

    // Headings: a dedicated bold family, falling back to the proportional set.
    let mut heading_chain = vec![BOLD_KEY.to_owned()];
    if let Some(prop) = fonts.families.get(&egui::FontFamily::Proportional) {
        heading_chain.extend(prop.iter().cloned());
    }
    fonts
        .families
        .insert(egui::FontFamily::Name(HEADING_FAMILY.into()), heading_chain);

    ctx.set_fonts(fonts);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Installing the embedded fonts must (a) parse without panicking and (b)
    /// make every Hangul character the UI actually shows resolve to a real
    /// glyph rather than the replacement box, in both the body and heading
    /// families. Because the fonts are embedded, this is deterministic on every
    /// platform and CI image.
    #[test]
    fn embedded_fonts_resolve_hangul_glyphs() {
        let ctx = egui::Context::default();
        install_korean_fonts(&ctx);

        // Force the font atlas to build and lay out Korean text; a malformed
        // font would panic here.
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading("한글 제목");
                ui.label("한글 본문 — 업스케일");
            });
        });

        let sample = "한글이미지업스케일모델다운로드출력폴더선택시작취소품질배율타일크기가로세로비율";
        let body = egui::FontId::proportional(16.0);
        let heading = egui::FontId::new(20.0, egui::FontFamily::Name(HEADING_FAMILY.into()));
        let (body_ok, heading_ok) =
            ctx.fonts(|f| (f.has_glyphs(&body, sample), f.has_glyphs(&heading, sample)));
        assert!(body_ok, "regular font failed to resolve Hangul");
        assert!(heading_ok, "bold/heading font failed to resolve Hangul");
    }
}
