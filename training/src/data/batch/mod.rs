pub mod cifar100;
pub mod imagenet1k;
pub mod mnist;

use burn::prelude::*;
use polars::prelude::*;

pub struct Batch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

pub trait FrameBatcher<B: Backend>: Send + Sync {
    fn batch(&self, df: DataFrame, device: &B::Device) -> Batch<B>;
}
