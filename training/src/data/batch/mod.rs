pub mod cifar100;
pub mod imagenet1k;
pub mod mnist;

use burn::prelude::*;
use polars::prelude::*;

pub trait FrameBatcher<B: Backend, O>: Send + Sync {
    fn batch(&self, df: DataFrame, device: &B::Device) -> O;
}
