use burn::{prelude::*, tensor::DType};
use polars::prelude::*;

use crate::{
    augmentations::Pipeline,
    data::batch::{Batch, FrameBatcher},
};

const IMAGECOL: &str = "image";
const LABELCOL: &str = "label";

pub struct Cifar10Batcher;

impl Cifar10Batcher {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl<B: Backend> FrameBatcher<B> for Cifar10Batcher {
    fn batch(&self, df: DataFrame, transforms: Arc<Pipeline<B>>, device: &B::Device) -> Batch<B> {
        let batch_size = df.height();

        // Image handling
        let total_images = batch_size * 32 * 32 * 3;

        let mut imagebuf: Vec<u8> = Vec::with_capacity(total_images);
        df.column(IMAGECOL)
            .unwrap()
            .binary()
            .unwrap()
            .into_no_null_iter()
            .for_each(|chunk| imagebuf.extend_from_slice(chunk));

        // Image handling
        let imagedata = TensorData::from_bytes_vec(imagebuf, [batch_size, 32, 32, 3], DType::U8)
            .convert_dtype(DType::F32);

        let images = transforms.execute(
            Tensor::<B, 4>::from_data(imagedata, device)
                .swap_dims(1, -1)
                .div_scalar(255),
        );

        // Label handling
        let labelbuf: Vec<i64> = df
            .column(LABELCOL)
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let labels = Tensor::<B, 1, Int>::from_ints(labelbuf.as_slice(), device);

        Batch {
            images,
            targets: labels,
        }
    }
}
