use burn::{prelude::*, tensor::DType};
use polars::prelude::*;

use crate::data::batch::{Batch, FrameBatcher};

const IMAGECOL: &str = "image";
const LABELCOL: &str = "label";

pub struct MnistBatcher;

impl MnistBatcher {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl<B: Backend> FrameBatcher<B> for MnistBatcher {
    fn batch(&self, df: DataFrame, device: &B::Device) -> Batch<B> {
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
        let imagedata = TensorData::from_bytes_vec(imagebuf, [batch_size, 1, 28, 28], DType::U8)
            .convert_dtype(DType::F32);
        let images = Tensor::<B, 4>::from_data(imagedata, device)
            .div_scalar(255)
            .sub_scalar(0.1307)
            .div_scalar(0.3081);

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
