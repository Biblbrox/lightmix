use polars::prelude::*;

use crate::data::mapper::{FrameMapper, decode_image_lazy};

const IMAGECOL: &str = "img";

pub struct Cifar100Mapper;

impl Cifar100Mapper {
    pub fn decoder() -> FrameMapper {
        Arc::new(|df| decode_image_lazy(df, IMAGECOL))
    }
}
