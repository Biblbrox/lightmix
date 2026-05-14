use std::sync::Arc;

use burn::{data::dataloader::DataLoader, tensor::backend::Backend};
use polars::prelude::*;

use crate::{
    augmentations::Pipeline,
    data::{
        batch::{FrameBatcher, ImageBatch},
        dataloader::{InMemoryDataLoader, StreamingDataLoader},
        mapper::LazyMapper,
        strategy::{FrameBatchStrategy, fixed::FixedBatchStrategy},
    },
};

pub struct StreamingDataLoaderBuilder<B: Backend> {
    batcher: Arc<dyn FrameBatcher<B>>,
    strategy: Option<Box<dyn FrameBatchStrategy>>,
    mapper: Option<LazyMapper>,
    transforms: Option<Arc<Pipeline<B>>>,
    device: Option<B::Device>,
}

impl<B: Backend> StreamingDataLoaderBuilder<B> {
    pub fn new(batcher: Arc<dyn FrameBatcher<B>>) -> Self {
        Self {
            batcher,
            strategy: None,
            mapper: None,
            transforms: None,
            device: None,
        }
    }

    pub fn with_strategy(mut self, strategy: impl FrameBatchStrategy + 'static) -> Self {
        self.strategy = Some(Box::new(strategy));
        self
    }

    pub fn with_mapper(mut self, mapper: LazyMapper) -> Self {
        self.mapper = Some(mapper);
        self
    }

    pub fn with_transforms(mut self, transforms: Arc<Pipeline<B>>) -> Self {
        self.transforms = Some(transforms);
        self
    }

    pub fn with_device(mut self, device: B::Device) -> Self {
        self.device = Some(device);
        self
    }

    pub fn build(self, dataset: LazyFrame) -> Arc<dyn DataLoader<B, ImageBatch<B>>> {
        Arc::new(StreamingDataLoader::new(
            dataset,
            self.batcher,
            self.strategy
                .unwrap_or(Box::new(FixedBatchStrategy::new(1))),
            self.transforms
                .unwrap_or(Arc::new(Pipeline::<B>::default())),
            self.device.unwrap_or_default(),
        ))
    }
}

pub struct InMemoryDataLoaderBuilder<B: Backend> {
    batcher: Arc<dyn FrameBatcher<B>>,
    transforms: Option<Arc<Pipeline<B>>>,
    batch_size: Option<usize>,
    num_workers: Option<usize>,
    device: Option<B::Device>,
}

impl<B: Backend> InMemoryDataLoaderBuilder<B> {
    pub fn new(batcher: Arc<dyn FrameBatcher<B>>) -> Self {
        Self {
            batcher,
            transforms: None,
            batch_size: None,
            num_workers: None,
            device: None,
        }
    }

    pub fn with_transforms(mut self, transforms: Arc<Pipeline<B>>) -> Self {
        self.transforms = Some(transforms);
        self
    }

    pub fn with_device(mut self, device: B::Device) -> Self {
        self.device = Some(device);
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    pub fn with_num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = Some(num_workers);
        self
    }

    pub fn build(self, dataset: LazyFrame) -> Arc<dyn DataLoader<B, ImageBatch<B>>> {
        Arc::new(InMemoryDataLoader::new(
            dataset,
            self.batcher,
            self.transforms
                .unwrap_or(Arc::new(Pipeline::<B>::default())),
            self.batch_size.unwrap_or(1),
            self.num_workers.unwrap_or(0),
            self.device.unwrap_or_default(),
        ))
    }
}
