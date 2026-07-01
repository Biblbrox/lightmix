pub mod modelnet40;

use std::sync::Arc;

use burn::tensor::{DType, Int, Shape, TensorData};
use burn::{prelude::Tensor, tensor::backend::Backend};
use polars::prelude::*;

use crate::augmentations::Pipeline;

/// Unified batch struct. Data stored as a flat Tensor<B, 1> and reshaped
/// by each model to the appropriate shape.
#[derive(Clone)]
pub struct Batch<B: Backend> {
    pub data: Tensor<B, 1>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batch<B> {
    pub fn shuffle(&self, seed: u64) -> Self {
        let b = self.targets.dims()[0];
        let device = self.targets.device();
        let mut indices: Vec<i32> = (0..b as i32).collect();
        let mut rng = fastrand::Rng::with_seed(seed);
        for i in (1..b as i32).rev() {
            let j = rng.usize(0..=(i as usize));
            indices.swap(i as usize, j);
        }
        let idx = Tensor::<B, 1, Int>::from_ints(indices.as_slice(), &device);
        Self {
            data: self.data.clone().select(0, idx.clone()),
            targets: self.targets.clone().select(0, idx),
        }
    }

    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            data: self.data.clone().to_device(device),
            targets: self.targets.clone().to_device(device),
        }
    }

    pub fn subbatch(&self, indices: &[usize]) -> Self {
        let idx_tensor = Tensor::<B, 1, Int>::from_ints(indices, &self.targets.device());
        Self {
            data: self.data.clone().select(0, idx_tensor.clone()),
            targets: self.targets.clone().select(0, idx_tensor),
        }
    }

    pub fn slice(&self, start: usize, end: usize, stride: usize) -> Self {
        let device = self.targets.device();
        let indices: Vec<i32> = (start as i32..end as i32).step_by(stride).collect();
        let idx_tensor = Tensor::<B, 1, Int>::from_ints(indices.as_slice(), &device);
        Self {
            data: self.data.clone().select(0, idx_tensor.clone()),
            targets: self.targets.clone().select(0, idx_tensor),
        }
    }

    pub fn batch_size(&self) -> usize {
        self.targets.dims()[0]
    }
}

/// Unified batcher trait for all data types.
pub trait Batcher<B: Backend>: Send + Sync {
    fn generic_batch(
        &self,
        df: DataFrame,
        transforms: Arc<Pipeline<B>>,
        shape: Shape,
        data_col: &str,
        label_col: &str,
        device: &B::Device,
    ) -> Batch<B> {
        let flat_size = shape.clone().flatten().len();
        let mut buf: Vec<u8> = Vec::with_capacity(flat_size);
        df.column(data_col)
            .unwrap()
            .binary()
            .unwrap()
            .iter()
            .flatten()
            .for_each(|chunk| buf.extend_from_slice(chunk));

        let data = TensorData::from_bytes_vec(buf, shape, DType::U8).convert_dtype(DType::F32);
        let labelbuf: Vec<i64> = df
            .column(label_col)
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();

        let labels = Tensor::<B, 1, Int>::from_ints(labelbuf.as_slice(), device);
        let transformed = transforms.execute(
            Tensor::<B, 4>::from_data(data, device)
                .swap_dims(1, -1)
                .div_scalar(255),
        );

        Batch {
            data: transformed.flatten::<1>(0, -1),
            targets: labels,
        }
    }

    fn batch(&self, df: DataFrame, transforms: Arc<Pipeline<B>>, device: &B::Device) -> Batch<B>;
}

macro_rules! define_image_batcher {
    ($name:ident, $width:expr, $height:expr, $channels:expr, $data_col:expr, $label_col:expr) => {
        pub struct $name;

        impl $name {
            pub fn new() -> Arc<Self> {
                Arc::new(Self)
            }
        }

        impl<B: Backend> Batcher<B> for $name {
            fn batch(
                &self,
                df: DataFrame,
                transforms: Arc<Pipeline<B>>,
                device: &B::Device,
            ) -> Batch<B> {
                let b = df.height();
                self.generic_batch(
                    df,
                    transforms,
                    Shape::new([b, $width, $height, $channels]),
                    $data_col,
                    $label_col,
                    device,
                )
            }
        }
    };
}

define_image_batcher!(Cifar10Batcher, 32, 32, 3, "image", "label");
define_image_batcher!(Cifar100Batcher, 32, 32, 3, "image", "label");
define_image_batcher!(FashionMnistBatcher, 28, 28, 1, "image", "label");
define_image_batcher!(Food101Batcher, 96, 96, 3, "image", "label");
define_image_batcher!(ImageNet1kBatcher, 224, 224, 3, "image", "label");
define_image_batcher!(MnistBatcher, 28, 28, 1, "image", "label");
define_image_batcher!(TinyImageNetBatcher, 64, 64, 3, "image", "label");
