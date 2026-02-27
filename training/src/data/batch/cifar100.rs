use burn::{prelude::*, tensor::DType};
use polars::prelude::*;

use crate::data::batch::FrameBatcher;

const IMAGECOL: &str = "img";
const FINE_LABELCOL: &str = "fine_label";
const COARSE_LABELCOL: &str = "coarse_label";

pub struct Cifar100Batch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub fine_targets: Tensor<B, 1, Int>,
    pub coarse_targets: Tensor<B, 1, Int>,
}

pub struct Cifar100Batcher;

impl Cifar100Batcher {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl<B: Backend> FrameBatcher<B, Cifar100Batch<B>> for Cifar100Batcher {
    fn batch(&self, df: DataFrame, device: &B::Device) -> Cifar100Batch<B> {
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
        let imagedata = TensorData::from_bytes_vec(imagebuf, [batch_size, 32, 32, 3], DType::U8)
            .convert_dtype(DType::F32);

        let mean = Tensor::<B, 1>::from_floats([0.5071, 0.4867, 0.4408], device)
            .reshape([1, 3, 1, 1])
            .expand([batch_size, 3, 32, 32]);

        let std = Tensor::<B, 1>::from_floats([0.2675, 0.2565, 0.2761], device)
            .reshape([1, 3, 1, 1])
            .expand([batch_size, 3, 32, 32]);

        let images = Tensor::<B, 4>::from_data(imagedata, device)
            .swap_dims(1, -1)
            .div_scalar(255)
            .sub(mean)
            .div(std);

        // Label handling
        let fine_labelbuf: Vec<i64> = df
            .column(FINE_LABELCOL)
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let fine_labels = Tensor::<B, 1, Int>::from_ints(fine_labelbuf.as_slice(), device);

        let coarse_labelbuf: Vec<i64> = df
            .column(COARSE_LABELCOL)
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let coarse_labels = Tensor::<B, 1, Int>::from_ints(coarse_labelbuf.as_slice(), device);

        Cifar100Batch {
            images,
            fine_targets: fine_labels,
            coarse_targets: coarse_labels,
        }
    }
}
