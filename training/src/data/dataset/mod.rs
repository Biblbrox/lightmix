pub mod cifar100;
pub mod imagenet1k;
pub mod mnist;

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
