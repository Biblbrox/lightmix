use polars::prelude::*;

use crate::data::mapper::{FrameMapper, decode_image_lazy};

pub struct MnistMapper;

const IMAGECOL: &str = "image";

impl MnistMapper {
    pub fn decoder() -> FrameMapper {
        Arc::new(|df| decode_image_lazy(df, IMAGECOL))
    }
}
