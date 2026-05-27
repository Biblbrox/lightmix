use polars::prelude::*;

use crate::data::mapper::{FrameMapper, decode_image_lazy};

pub struct FashionMnistMapper;

const IMAGECOL: &str = "image";

impl FashionMnistMapper {
    pub fn decoder() -> FrameMapper {
        Arc::new(|df| decode_image_lazy(df, IMAGECOL))
    }
}
