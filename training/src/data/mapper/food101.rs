use polars::prelude::*;

use crate::data::mapper::{FrameMapper, decode_image_lazy};

const IMAGECOL: &str = "image";

pub struct Food101Mapper;

impl Food101Mapper {
    pub fn decoder() -> FrameMapper {
        Arc::new(|df| decode_image_lazy(df, IMAGECOL))
    }
}
