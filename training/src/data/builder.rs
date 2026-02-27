use burn::{data::dataloader::DataLoader, prelude::*};
use polars::prelude::*;

use crate::data::{
    batch::FrameBatcher,
    dataloader::StreamingDataLoader,
    mapper::{FrameMapper, LazyMapper},
};

pub struct StreamingDataLoaderBuilder<B: Backend, O> {
    batcher: Arc<dyn FrameBatcher<B, O>>,
    batch_size: Option<usize>,
    shuffle: Option<u64>,
    batch_mapper: Option<FrameMapper>,
    dataset_mapper: Option<LazyMapper>,
    device: Option<B::Device>,
}

impl<B: Backend, O: Send + Sync + 'static> StreamingDataLoaderBuilder<B, O> {
    pub fn new(batcher: Arc<dyn FrameBatcher<B, O>>) -> Self {
        Self {
            batcher,
            batch_mapper: None,
            dataset_mapper: None,
            batch_size: None,
            shuffle: None,
            device: None,
        }
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    pub fn with_shuffle(mut self, seed: u64) -> Self {
        self.shuffle = Some(seed);
        self
    }

    pub fn with_batch_mapper(mut self, batch_mapper: FrameMapper) -> Self {
        self.batch_mapper = Some(batch_mapper);
        self
    }

    pub fn with_dataset_mapper(mut self, dataset_mapper: LazyMapper) -> Self {
        self.dataset_mapper = Some(dataset_mapper);
        self
    }

    pub fn with_device(mut self, device: B::Device) -> Self {
        self.device = Some(device);
        self
    }

    pub fn build(self, dataset: LazyFrame) -> Arc<dyn DataLoader<B, O>> {
        Arc::new(StreamingDataLoader::new(
            dataset,
            self.batcher,
            self.batch_mapper,
            self.batch_size.unwrap_or(1),
            self.shuffle,
            self.device.unwrap_or_default(),
        ))
    }
}
