/// Clamps a requested surface size into the range the device can present.
///
/// Surfaces must be at least 1x1 (a minimized window reports 0x0) and no larger
/// than the device's `max_texture_dimension_2d`.
pub fn clamp_surface_size(width: u32, height: u32, max_dimension: u32) -> (u32, u32) {
    let max = max_dimension.max(1);
    (width.clamp(1, max), height.clamp(1, max))
}

#[cfg(test)]
mod tests {
    use super::clamp_surface_size;

    #[test]
    fn passes_through_valid_sizes() {
        assert_eq!(clamp_surface_size(1280, 720, 8192), (1280, 720));
    }

    #[test]
    fn zero_size_becomes_one_pixel() {
        assert_eq!(clamp_surface_size(0, 0, 8192), (1, 1));
    }

    #[test]
    fn oversized_is_limited_to_device_max() {
        assert_eq!(clamp_surface_size(10_000, 300, 4096), (4096, 300));
    }

    #[test]
    fn zero_device_limit_does_not_panic() {
        assert_eq!(clamp_surface_size(800, 600, 0), (1, 1));
    }
}
