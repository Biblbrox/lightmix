use burn::{prelude::*, tensor::DType};
use polars::prelude::*;

use crate::{
    augmentations::Pipeline,
    data::batch::{FrameBatcher, ImageBatch},
};

const IMAGECOL: &str = "image";
const LABELCOL: &str = "label";

pub struct FashionMnistBatcher;

impl FashionMnistBatcher {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl<B: Backend> FrameBatcher<B> for FashionMnistBatcher {
    fn batch(
        &self,
        df: DataFrame,
        transforms: Arc<Pipeline<B>>,
        device: &B::Device,
    ) -> ImageBatch<B> {
        let batch_size = df.height();

        // Image handling
        let imagebuf = df
            .column(IMAGECOL)
            .unwrap()
            .binary()
            .unwrap()
            .into_no_null_iter()
            .flatten()
            .copied()
            .collect();
        let imagedata = TensorData::from_bytes_vec(imagebuf, [batch_size, 28, 28, 1], DType::U8)
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

        ImageBatch {
            images,
            targets: labels,
        }
    }
}
