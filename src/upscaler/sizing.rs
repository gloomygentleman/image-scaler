//! Target output size: by scale factor, or by explicit pixel dimensions.
//!
//! The AI model upscales by its own fixed factor (e.g. ×4). To let the user ask
//! for an arbitrary factor or exact dimensions, the pipeline runs the model once
//! and then resamples the super-resolved image to the size resolved here. This
//! keeps the AI-recovered detail while hitting any requested size.

/// How the user wants the output sized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TargetSize {
    /// Multiply both source dimensions by this factor (applied per image).
    Scale(f32),
    /// Fit within `w` × `h`, preserving each image's aspect ratio (no distortion).
    FitBox { w: u32, h: u32 },
    /// Force exactly `w` × `h` (aspect ratio may change).
    Exact { w: u32, h: u32 },
}

impl TargetSize {
    /// Resolve the output dimensions for a `src_w` × `src_h` source.
    ///
    /// The result is always at least `1 × 1`.
    pub fn resolve(self, src_w: u32, src_h: u32) -> (u32, u32) {
        let sw = src_w.max(1) as f32;
        let sh = src_h.max(1) as f32;
        match self {
            TargetSize::Scale(f) => {
                let f = if f.is_finite() { f.max(0.01) } else { 1.0 };
                ((sw * f).round().max(1.0) as u32, (sh * f).round().max(1.0) as u32)
            }
            TargetSize::Exact { w, h } => (w.max(1), h.max(1)),
            TargetSize::FitBox { w, h } => {
                let s = (w.max(1) as f32 / sw).min(h.max(1) as f32 / sh);
                ((sw * s).round().max(1.0) as u32, (sh * s).round().max(1.0) as u32)
            }
        }
    }

    /// Effective magnification relative to the source, for display.
    pub fn factor_for(self, src_w: u32, src_h: u32) -> f32 {
        let (w, _) = self.resolve(src_w, src_h);
        w as f32 / src_w.max(1) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_multiplies_both_dimensions() {
        assert_eq!(TargetSize::Scale(4.0).resolve(100, 50), (400, 200));
        assert_eq!(TargetSize::Scale(2.5).resolve(100, 100), (250, 250));
    }

    #[test]
    fn exact_forces_dimensions() {
        assert_eq!(TargetSize::Exact { w: 1920, h: 1080 }.resolve(640, 480), (1920, 1080));
    }

    #[test]
    fn fit_box_preserves_aspect_ratio() {
        // 800x600 (4:3) into a 1000x1000 box -> limited by width-ish; 4:3 keeps.
        let (w, h) = TargetSize::FitBox { w: 1000, h: 1000 }.resolve(800, 600);
        assert_eq!((w, h), (1000, 750));
        // Tall source into a wide box -> limited by height.
        let (w, h) = TargetSize::FitBox { w: 1000, h: 400 }.resolve(500, 1000);
        assert_eq!((w, h), (200, 400));
    }

    #[test]
    fn never_returns_zero() {
        assert_eq!(TargetSize::Scale(0.0).resolve(0, 0), (1, 1));
        assert_eq!(TargetSize::Scale(f32::NAN).resolve(10, 10), (10, 10));
    }

    #[test]
    fn factor_matches_resolved_width() {
        assert!((TargetSize::Scale(3.0).factor_for(120, 80) - 3.0).abs() < 1e-6);
        assert!((TargetSize::Exact { w: 240, h: 160 }.factor_for(120, 80) - 2.0).abs() < 1e-6);
    }
}
