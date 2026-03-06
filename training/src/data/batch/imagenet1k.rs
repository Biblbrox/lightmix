use burn::{
    prelude::*,
    tensor::{DType, Transaction},
};
use cubecl::cpu::CpuDevice;
use polars::prelude::*;

use crate::data::batch::FrameBatcher;

const IMAGECOL: &str = "image";
const LABELCOL: &str = "label";

pub struct ImageNet1kBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

pub struct ImageNet1kBatcher;

impl ImageNet1kBatcher {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl<B: Backend> FrameBatcher<B, ImageNet1kBatch<B>> for ImageNet1kBatcher {
    fn batch(&self, df: DataFrame, device: &B::Device) -> ImageNet1kBatch<B> {
        let batch_size = df.height();

        // Image handling
        let total_images = batch_size * 224 * 224 * 3;

        let mut imagebuf: Vec<u8> = Vec::with_capacity(total_images);
        df.column(IMAGECOL)
            .unwrap()
            .binary()
            .unwrap()
            .into_no_null_iter()
            .for_each(|chunk| imagebuf.extend_from_slice(chunk));

        // Label handling
        let labelbuf: Vec<i64> = df
            .column(LABELCOL)
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();

        let imagedata = TensorData::from_bytes_vec(imagebuf, [batch_size, 224, 224, 3], DType::U8)
            .convert_dtype(DType::F32);

        let mean = Tensor::<B, 1>::from_floats([0.485, 0.456, 0.406], device)
            .reshape([1, 3, 1, 1])
            .expand([batch_size, 3, 224, 224]);

        let std = Tensor::<B, 1>::from_floats([0.229, 0.224, 0.225], device)
            .reshape([1, 3, 1, 1])
            .expand([batch_size, 3, 224, 224]);

        let images = Tensor::<B, 4>::from_data(imagedata, device)
            .swap_dims(1, -1)
            .div_scalar(255)
            .sub(mean)
            .div(std);

        let labels = Tensor::<B, 1, Int>::from_ints(labelbuf.as_slice(), device);

        ImageNet1kBatch {
            images: images,
            targets: labels,
        }
    }
}
