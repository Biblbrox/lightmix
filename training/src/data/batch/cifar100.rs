use std::sync::Arc;

use burn::tensor::backend::Backend;
use polars::prelude::*;

use crate::{
    augmentations::Pipeline,
    data::batch::{Batch, Batcher},
};

const IMAGECOL: &str = "image";
const LABELCOL: &str = "fine_label";

pub struct Cifar100Batcher;

impl Cifar100Batcher {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl<B: Backend> Batcher<B> for Cifar100Batcher {
    fn batch(&self, df: DataFrame, transforms: Arc<Pipeline<B>>, device: &B::Device) -> Batch<B> {
        self.image_batch(df, transforms, 32, 32, IMAGECOL, LABELCOL, 3, device)
    }
}
