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

#[derive(Clone)]
pub struct ImageBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> ImageBatch<B> {
    pub fn shuffle(&self, seed: u64) -> Self {
        let b = self.targets.dims()[0];
        let device = self.targets.device();

        let mut indices: Vec<i32> = (0..b as i32).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);

        let idx = Tensor::<B, 1, Int>::from_ints(indices.as_slice(), &device);

        Self {
            images: self.images.clone().select(0, idx.clone()),
            targets: self.targets.clone().select(0, idx),
        }
    }

    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            images: self.images.clone().to_device(device),
            targets: self.targets.clone().to_device(device),
        }
    }

    pub fn subbatch(&self, indices: &[usize]) -> Self {
        let idx_tensor = Tensor::<B, 1, Int>::from_ints(indices, &self.targets.device());

        Self {
            images: self.images.clone().select(0, idx_tensor.clone()),
            targets: self.targets.clone().select(0, idx_tensor),
        }
    }

    pub fn slice(&self, start: usize, end: usize, stride: usize) -> Self {
        let device = self.targets.device();
        let indices: Vec<i32> = (start as i32..end as i32).step_by(stride).collect();
        let idx_tensor = Tensor::<B, 1, Int>::from_ints(indices.as_slice(), &device);

        Self {
            images: self.images.clone().select(0, idx_tensor.clone()),
            targets: self.targets.clone().select(0, idx_tensor),
        }
    }
}

pub trait FrameBatcher<B: Backend>: Send + Sync {
    fn batch(
        &self,
        df: DataFrame,
        transforms: Arc<Pipeline<B>>,
        device: &B::Device,
    ) -> ImageBatch<B>;
}

#[derive(Clone)]
pub struct CloudBatch<B: Backend> {
    pub points: Tensor<B, 3>,       // [B, N, C]
    pub targets: Tensor<B, 1, Int>, // [B]
}

impl<B: Backend> CloudBatch<B> {
    pub fn shuffle(&self, seed: u64) -> Self {
        let b = self.targets.dims()[0];
        let device = self.targets.device();
        let mut indices: Vec<i32> = (0..b as i32).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
        let idx = Tensor::<B, 1, Int>::from_ints(indices.as_slice(), &device);
        Self {
            points: self.points.clone().select(0, idx.clone()),
            targets: self.targets.clone().select(0, idx),
        }
    }

    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            points: self.points.clone().to_device(device),
            targets: self.targets.clone().to_device(device),
        }
    }

    pub fn subbatch(&self, indices: &[usize]) -> Self {
        let idx_tensor = Tensor::<B, 1, Int>::from_ints(indices, &self.targets.device());
        Self {
            points: self.points.clone().select(0, idx_tensor.clone()),
            targets: self.targets.clone().select(0, idx_tensor),
        }
    }

    pub fn slice(&self, start: usize, end: usize, stride: usize) -> Self {
        let device = self.targets.device();
        let indices: Vec<i32> = (start as i32..end as i32).step_by(stride).collect();
        let idx_tensor = Tensor::<B, 1, Int>::from_ints(indices.as_slice(), &device);
        Self {
            points: self.points.clone().select(0, idx_tensor.clone()),
            targets: self.targets.clone().select(0, idx_tensor),
        }
    }
}

pub trait CloudBatcher<B: Backend>: Send + Sync {
    fn batch(&self, df: DataFrame, device: &B::Device) -> CloudBatch<B>;
}
