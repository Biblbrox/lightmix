pub mod cifar10;
pub mod cifar100;
pub mod fashionmnist;
pub mod food101;
pub mod imagenet1k;
pub mod mnist;
pub mod modelnet40;
pub mod tinyimagenet;

use polars::prelude::*;

pub enum LazyFiletype {
    Parquet,
    Arrow,
    Csv,
}

pub trait LazyDataset {
    fn train(&self) -> LazyFrame;
    fn validation(&self) -> LazyFrame;
    fn test(&self) -> LazyFrame;
}
