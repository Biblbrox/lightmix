use std::{marker::PhantomData, num::NonZero, sync::Arc};

use burn::{
    data::dataloader::{DataLoader, DataLoaderIterator, Progress},
    prelude::*,
};
use polars::prelude::*;

use crate::data::{batch::FrameBatcher, mapper::FrameMapper};

pub struct StreamingDataLoader<B: Backend, O> {
    dataset: LazyFrame,
    batcher: Arc<dyn FrameBatcher<B, O>>,
    batch_mapper: Option<FrameMapper>,
    batch_size: usize,
    shuffle: Option<u64>,
    total_items: usize,
    device: B::Device,
    _o: PhantomData<O>,
}

impl<B: Backend, O> Clone for StreamingDataLoader<B, O> {
    fn clone(&self) -> Self {
        Self {
            dataset: self.dataset.clone(),
            batcher: self.batcher.clone(),
            batch_mapper: self.batch_mapper.clone(),
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            total_items: self.total_items,
            device: self.device.clone(),
            _o: PhantomData,
        }
    }
}

impl<B: Backend, O> StreamingDataLoader<B, O> {
    pub fn new(
        dataset: impl Into<LazyFrame>,
        batcher: Arc<dyn FrameBatcher<B, O>>,
        batch_mapper: Option<FrameMapper>,
        batch_size: usize,
        shuffle: Option<u64>,
        device: B::Device,
    ) -> Self {
        let dataset = dataset.into();
        let total_items = dataset
            .clone()
            .select([len()])
            .collect_with_engine(Engine::Streaming)
            .unwrap()
            .column("len")
            .unwrap()
            .u32()
            .unwrap()
            .get(0)
            .unwrap() as usize;
        Self {
            dataset,
            batcher,
            batch_mapper,
            batch_size,
            shuffle,
            total_items,
            device,
            _o: PhantomData,
        }
    }
}

impl<B, O> DataLoader<B, O> for StreamingDataLoader<B, O>
where
    B: Backend,
    O: Send + Sync + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        Box::new(StreamingDataLoaderIterator::new(
            self.dataset.clone(),
            self.batcher.clone(),
            self.batch_mapper.clone(),
            self.batch_size,
            self.shuffle,
            self.total_items,
            self.device.clone(),
        ))
    }

    fn num_items(&self) -> usize {
        self.total_items
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, O>> {
        Arc::new(StreamingDataLoader::new(
            self.dataset.clone(),
            self.batcher.clone(),
            self.batch_mapper.clone(),
            self.batch_size,
            self.shuffle,
            device.clone(),
        ))
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, O>> {
        Arc::new(StreamingDataLoader::new(
            self.dataset
                .clone()
                .slice(start as i64, (end - start) as u32),
            self.batcher.clone(),
            self.batch_mapper.clone(),
            self.batch_size,
            self.shuffle,
            self.device.clone(),
        ))
    }
}

/// Basically a tracking iterator over DataFrame batches with a method to calculate progress
pub struct StreamingDataLoaderIterator<B: Backend, O> {
    batches: CollectBatches,
    batcher: Arc<dyn FrameBatcher<B, O>>,
    batch_mapper: Option<FrameMapper>,
    batch_size: usize,
    shuffle: Option<u64>,
    current_batch: usize,
    total_items: usize,
    device: B::Device,
    _p: PhantomData<O>,
}

impl<B: Backend, O> StreamingDataLoaderIterator<B, O> {
    pub fn new(
        dataset: LazyFrame,
        batcher: Arc<dyn FrameBatcher<B, O>>,
        batch_mapper: Option<FrameMapper>,
        batch_size: usize,
        shuffle: Option<u64>,
        total_items: usize,
        device: B::Device,
    ) -> Self {
        Self {
            batches: dataset
                .collect_batches(
                    Engine::Auto,
                    shuffle.is_none(),
                    NonZero::new(batch_size),
                    true,
                )
                .unwrap(),
            batcher,
            batch_mapper,
            batch_size,
            shuffle,
            current_batch: 0,
            total_items,
            device,
            _p: PhantomData,
        }
    }
}

impl<B: Backend, O> Iterator for StreamingDataLoaderIterator<B, O> {
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        self.current_batch += 1;
        if let Some(mut batch) = self.batches.next().transpose().ok().flatten() {
            if let Some(seed) = self.shuffle {
                batch = batch
                    .sample_n_literal(batch.height(), false, true, Some(seed))
                    .unwrap();
            }

            if let Some(ref map) = self.batch_mapper {
                batch = map(batch);
            }

            return Some(self.batcher.batch(batch, &self.device));
        }

        None
    }
}

impl<B: Backend, O> DataLoaderIterator<O> for StreamingDataLoaderIterator<B, O> {
    fn progress(&self) -> Progress {
        Progress::new(self.current_batch * self.batch_size, self.total_items)
    }
}
