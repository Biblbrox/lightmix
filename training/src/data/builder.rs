use burn::{data::dataloader::DataLoader, prelude::*};
use polars::prelude::*;

use crate::data::{
    batch::{Batch, FrameBatcher},
    dataloader::StreamingDataLoader,
    mapper::LazyMapper,
    strategy::{FrameBatchStrategy, fixed::FixedBatchStrategy},
};

pub struct StreamingDataLoaderBuilder<B: Backend> {
    batcher: Arc<dyn FrameBatcher<B>>,
    strategy: Option<Box<dyn FrameBatchStrategy>>,
    mapper: Option<LazyMapper>,
    device: Option<B::Device>,
}

impl<B: Backend> StreamingDataLoaderBuilder<B> {
    pub fn new(batcher: Arc<dyn FrameBatcher<B>>) -> Self {
        Self {
            batcher,
            strategy: None,
            mapper: None,
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

    pub fn with_device(mut self, device: B::Device) -> Self {
        self.device = Some(device);
        self
    }

    pub fn build(self, dataset: LazyFrame) -> Arc<dyn DataLoader<B, Batch<B>>> {
        Arc::new(StreamingDataLoader::new(
            dataset,
            self.batcher,
            self.strategy
                .unwrap_or(Box::new(FixedBatchStrategy::new(1))),
            self.device.unwrap_or_default(),
        ))
    }
}
