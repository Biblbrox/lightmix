pub mod cifar100;
pub mod imagenet1k;
pub mod mnist;
pub mod fashionmnist;
pub mod tinyimagenet;

use burn::prelude::*;
use polars::prelude::*;

use crate::augmentations::Pipeline;

pub struct Batch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

pub trait FrameBatcher<B: Backend>: Send + Sync {
    fn batch(&self, df: DataFrame, transforms: Arc<Pipeline<B>>, device: &B::Device) -> Batch<B>;
}
