//! Environment images: equirectangular Radiance `.hdr` files (as Poly Haven
//! ships them), decoded to linear RGB.

use std::fmt;
use std::io::Cursor;

/// Widest equirect accepted (its height is half this).
pub const MAX_HDR_WIDTH: u32 = 8192;

/// A decoded equirectangular image: rows top (straight up) to bottom
/// (straight down), each spanning a full turn around the horizon.
#[derive(Debug, Clone, PartialEq)]
pub struct HdrImage {
    pub width: u32,
    pub height: u32,
    /// Linear RGB radiance, `width × height` texels, row by row. Never
    /// negative or NaN.
    pub pixels: Vec<[f32; 3]>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HdrError {
    /// Not a Radiance file, or a broken one.
    Decode(String),
    /// Wider than [`MAX_HDR_WIDTH`].
    TooLarge { max: u32 },
    /// Not twice as wide as tall.
    NotEquirect { width: u32, height: u32 },
}

impl fmt::Display for HdrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(e) => write!(f, "not a readable .hdr file: {e}"),
            Self::TooLarge { max } => write!(f, "wider than {max} texels"),
            Self::NotEquirect { width, height } => write!(
                f,
                "{width}×{height} is not an equirectangular (2:1) environment"
            ),
        }
    }
}

impl std::error::Error for HdrError {}

/// Decodes a Radiance `.hdr` equirect.
pub fn load_hdr(bytes: &[u8]) -> Result<HdrImage, HdrError> {
    let mut reader = image::ImageReader::with_format(Cursor::new(bytes), image::ImageFormat::Hdr);
    let mut caps = image::Limits::default();
    caps.max_image_width = Some(MAX_HDR_WIDTH);
    caps.max_image_height = Some(MAX_HDR_WIDTH / 2);
    reader.limits(caps);
    let decoded = reader.decode().map_err(|e| match e {
        image::ImageError::Limits(_) => HdrError::TooLarge { max: MAX_HDR_WIDTH },
        e => HdrError::Decode(e.to_string()),
    })?;
    let rgb = decoded.into_rgb32f();
    let (width, height) = rgb.dimensions();
    if width != height * 2 || height == 0 {
        return Err(HdrError::NotEquirect { width, height });
    }
    let clean = |v: f32| if v.is_finite() { v.max(0.0) } else { 0.0 };
    let pixels = rgb
        .pixels()
        .map(|p| [clean(p.0[0]), clean(p.0[1]), clean(p.0[2])])
        .collect();
    Ok(HdrImage {
        width,
        height,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLOUDS: &[u8] =
        include_bytes!("../../../assets/env/kloofendal_48d_partly_cloudy_puresky_1k.hdr");
    const SUNSET: &[u8] = include_bytes!("../../../assets/env/belfast_sunset_puresky_1k.hdr");

    #[test]
    fn bundled_skies_decode_to_bright_equirects() {
        for (bytes, peak) in [(CLOUDS, 10_000.0), (SUNSET, 20.0)] {
            let image = load_hdr(bytes).unwrap();
            assert_eq!((image.width, image.height), (1024, 512));
            assert_eq!(image.pixels.len(), 1024 * 512);
            let max = image
                .pixels
                .iter()
                .map(|p| p[0].max(p[1]).max(p[2]))
                .fold(0.0, f32::max);
            assert!(max > peak, "peak {max}");
        }
    }

    #[test]
    fn garbage_and_wrong_shapes_are_refused() {
        assert!(matches!(
            load_hdr(b"not an image"),
            Err(HdrError::Decode(_))
        ));
        assert!(matches!(load_hdr(&[]), Err(HdrError::Decode(_))));
        // A valid Radiance file, but square.
        let mut square = Vec::new();
        image::Rgb32FImage::from_pixel(4, 4, image::Rgb([1.0, 0.5, 0.25]))
            .write_to(&mut Cursor::new(&mut square), image::ImageFormat::Hdr)
            .unwrap();
        assert_eq!(
            load_hdr(&square),
            Err(HdrError::NotEquirect {
                width: 4,
                height: 4
            })
        );
    }

    #[test]
    fn values_round_trip_and_oversized_files_are_refused() {
        let mut small = Vec::new();
        image::Rgb32FImage::from_fn(8, 4, |x, _| image::Rgb([x as f32, 2.0, 0.5]))
            .write_to(&mut Cursor::new(&mut small), image::ImageFormat::Hdr)
            .unwrap();
        let image = load_hdr(&small).unwrap();
        assert_eq!(image.pixels[3], [3.0, 2.0, 0.5]);
        // The header alone declares the size: no need to build a huge file.
        let huge = format!(
            "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {} +X {}\n",
            MAX_HDR_WIDTH,
            MAX_HDR_WIDTH * 2
        );
        assert_eq!(
            load_hdr(huge.as_bytes()),
            Err(HdrError::TooLarge { max: MAX_HDR_WIDTH })
        );
    }
}
