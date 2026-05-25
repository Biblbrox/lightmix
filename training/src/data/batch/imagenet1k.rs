use std::sync::Arc;

use burn::{
    prelude::Tensor,
    tensor::{DType, Int, TensorData, backend::Backend},
};
use polars::prelude::*;

use crate::{
    augmentations::Pipeline,
    data::batch::{Batch, Batcher},
};

const IMAGECOL: &str = "image";
const LABELCOL: &str = "label";

pub struct ImageNet1kBatcher;

impl ImageNet1kBatcher {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl<B: Backend> Batcher<B> for ImageNet1kBatcher {
    fn batch(&self, df: DataFrame, transforms: Arc<Pipeline<B>>, device: &B::Device) -> Batch<B> {
        let batch_size = df.height();

        let total_images = batch_size * 224 * 224 * 3;

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

        let imagedata = TensorData::from_bytes_vec(imagebuf, [batch_size, 224, 224, 3], DType::U8)
            .convert_dtype(DType::F32);
        let labels = Tensor::<B, 1, Int>::from_ints(labelbuf.as_slice(), device);

        let images = transforms.execute(
            Tensor::<B, 4>::from_data(imagedata, device)
                .swap_dims(1, -1)
                .div_scalar(255),
        );

        Batch {
            data: images.flatten::<1>(0, -1),
            targets: labels,
        }
    }
}
