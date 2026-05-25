pub mod cifar10;
pub mod cifar100;
pub mod fashionmnist;
pub mod food101;
pub mod imagenet1k;
pub mod mnist;
pub mod modelnet40;
pub mod tinyimagenet;

use std::sync::Arc;

use burn::tensor::Int;
use burn::{prelude::Tensor, tensor::backend::Backend};
use polars::prelude::*;
use rand::{SeedableRng, seq::SliceRandom};

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
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
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
    fn batch(&self, df: DataFrame, transforms: Arc<Pipeline<B>>, device: &B::Device) -> Batch<B>;
}
