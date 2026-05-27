use polars::prelude::*;

use crate::data::mapper::{FrameMapper, decode_image_lazy};

const IMAGECOL: &str = "image";

pub struct ImageNet1kMapper;

impl ImageNet1kMapper {
    pub fn decoder() -> FrameMapper {
        Arc::new(|df| decode_image_lazy(df, IMAGECOL))
    }
}
