use std::sync::Arc;

use crate::data::mapper::{FrameMapper, decode_image_lazy};

const IMAGECOL: &str = "image";

pub struct TinyImageNetMapper;

impl TinyImageNetMapper {
    pub fn decoder() -> FrameMapper {
        Arc::new(|df| decode_image_lazy(df, IMAGECOL))
    }
}
