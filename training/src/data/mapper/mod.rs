pub mod cifar100;
pub mod imagenet1k;
pub mod mnist;
pub mod fashionmnist;
pub mod tinyimagenet;

use polars::prelude::*;

pub type FrameMapper = Arc<dyn Fn(DataFrame) -> DataFrame + Sync + Send + 'static>;
pub type LazyMapper = Arc<dyn Fn(LazyFrame) -> LazyFrame + Sync + Send + 'static>;
