use std::sync::Arc;

use burn::{
    prelude::Tensor,
    tensor::{DType, Int, TensorData, backend::Backend},
};
use polars::prelude::*;

use crate::{
    augmentations::Pipeline,
    data::batch::{FrameBatcher, ImageBatch},
};

const IMAGECOL: &str = "image";
const LABELCOL: &str = "label";

pub struct Food101Batcher;

impl Food101Batcher {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl<B: Backend> FrameBatcher<B> for Food101Batcher {
    fn batch(
        &self,
        df: DataFrame,
        transforms: Arc<Pipeline<B>>,
        device: &B::Device,
    ) -> ImageBatch<B> {
        let batch_size = df.height();

        // Image handling
        let total_images = batch_size * 96 * 96 * 3;

        let mut imagebuf: Vec<u8> = Vec::with_capacity(total_images);
        df.column(IMAGECOL)
            .unwrap()
            .binary()
            .unwrap()
            .into_no_null_iter()
            .for_each(|chunk| imagebuf.extend_from_slice(chunk));

        let labelbuf: Vec<i64> = df
            .column(LABELCOL)
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();

        let imagedata = TensorData::from_bytes_vec(imagebuf, [batch_size, 96, 96, 3], DType::U8)
            .convert_dtype(DType::F32);

        let images = transforms.execute(
            Tensor::<B, 4>::from_data(imagedata, device)
                .permute([0, 3, 1, 2]) // [B,H,W,C] → [B,C,H,W]
                .div_scalar(255),
        );

        let labels = Tensor::<B, 1, Int>::from_ints(labelbuf.as_slice(), device);

        ImageBatch {
            images,
            targets: labels,
        }
    }
}
