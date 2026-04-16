use std::{
    num::NonZero,
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver},
    },
    thread,
};

use burn::{
    data::dataloader::{DataLoader, DataLoaderIterator, Progress},
    prelude::*,
};
use polars::prelude::*;

use crate::{
    augmentations::Pipeline,
    data::{
        batch::{Batch, FrameBatcher},
        strategy::FrameBatchStrategy,
    },
};

pub struct StreamingDataLoader<B: Backend> {
    dataset: LazyFrame,
    batcher: Arc<dyn FrameBatcher<B>>,
    strategy: Box<dyn FrameBatchStrategy>,
    transforms: Arc<Pipeline<B>>,
    total_items: usize,
    device: B::Device,
}

impl<B: Backend> Clone for StreamingDataLoader<B> {
    fn clone(&self) -> Self {
        Self {
            dataset: self.dataset.clone(),
            batcher: self.batcher.clone(),
            strategy: self.strategy.clone_dyn(),
            transforms: self.transforms.clone(),
            total_items: self.total_items,
            device: self.device.clone(),
        }
    }
}

impl<B: Backend> StreamingDataLoader<B> {
    pub fn new(
        dataset: impl Into<LazyFrame>,
        batcher: Arc<dyn FrameBatcher<B>>,
        strategy: Box<dyn FrameBatchStrategy>,
        transforms: Arc<Pipeline<B>>,
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
            strategy,
            transforms,
            total_items,
            device,
        }
    }
}

impl<B> DataLoader<B, Batch<B>> for StreamingDataLoader<B>
where
    B: Backend,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<Batch<B>> + 'a> {
        Box::new(StreamingDataLoaderIterator::new(
            self.dataset.clone(),
            self.batcher.clone(),
            self.strategy.clone_dyn(),
            self.transforms.clone(),
            self.total_items,
            self.device.clone(),
        ))
    }

    fn num_items(&self) -> usize {
        self.total_items
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, Batch<B>>> {
        Arc::new(StreamingDataLoader::new(
            self.dataset.clone(),
            self.batcher.clone(),
            self.strategy.clone_dyn(),
            self.transforms.clone(),
            device.clone(),
        ))
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, Batch<B>>> {
        Arc::new(StreamingDataLoader::new(
            self.dataset
                .clone()
                .slice(start as i64, (end - start) as u32),
            self.batcher.clone(),
            self.strategy.clone_dyn(),
            self.transforms.clone(),
            self.device.clone(),
        ))
    }
}

/// Basically a tracking iterator over DataFrame batches with a method to calculate progress
pub struct StreamingDataLoaderIterator<B: Backend> {
    batcher: Arc<dyn FrameBatcher<B>>,
    strategy: Box<dyn FrameBatchStrategy>,
    current_batch: usize,
    transforms: Arc<Pipeline<B>>,
    total_items: usize,
    device: B::Device,
}

impl<B: Backend> StreamingDataLoaderIterator<B> {
    pub fn new(
        dataset: LazyFrame,
        batcher: Arc<dyn FrameBatcher<B>>,
        mut strategy: Box<dyn FrameBatchStrategy>,
        transforms: Arc<Pipeline<B>>,
        total_items: usize,
        device: B::Device,
    ) -> Self {
        let stream = dataset
            .collect_batches(
                Engine::Auto,
                false,
                NonZero::new(strategy.chunk_size()),
                true,
            )
            .unwrap();
        strategy.init(stream);

        Self {
            batcher,
            strategy,
            current_batch: 0,
            transforms,
            total_items,
            device,
        }
    }
}

impl<B: Backend> Iterator for StreamingDataLoaderIterator<B> {
    type Item = Batch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        self.current_batch += 1;

        if let Some(df) = self.strategy.batch() {
            return Some(
                self.batcher
                    .batch(df, self.transforms.clone(), &self.device),
            );
        }

        None
    }
}

impl<B: Backend> DataLoaderIterator<Batch<B>> for StreamingDataLoaderIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(
            self.current_batch * self.strategy.batch_size(),
            self.total_items,
        )
    }
}

///////////////////////////////////////////////////////////////////////////////
// IN-MEMORY DATALOADER SECTION
///////////////////////////////////////////////////////////////////////////////
pub struct InMemoryDataLoader<B: Backend> {
    lf: LazyFrame,
    batcher: Arc<dyn FrameBatcher<B>>,
    transforms: Arc<Pipeline<B>>,
    items_total: usize,
    batch_size: usize,
    num_workers: usize,
    device: B::Device,
}

impl<B: Backend> InMemoryDataLoader<B> {
    pub fn new(
        lf: impl Into<LazyFrame>,
        batcher: Arc<dyn FrameBatcher<B>>,
        transforms: Arc<Pipeline<B>>,
        batch_size: usize,
        num_workers: usize,
        device: B::Device,
    ) -> Self {
        let lf = lf.into();
        let items_total = lf
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
            lf,
            batcher,
            transforms,
            items_total,
            batch_size,
            device,
            num_workers,
        }
    }
}

impl<B: Backend> DataLoader<B, Batch<B>> for InMemoryDataLoader<B> {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<Batch<B>> + 'a> {
        if self.num_workers > 0 {
            let (tx, rx) = mpsc::sync_channel(4 * self.num_workers);
            for idx in 0..self.num_workers {
                let num_workers = self.num_workers;
                let batcher = self.batcher.clone();
                let transforms = self.transforms.clone();
                let lf = self.lf.clone();
                let batch_size = self.batch_size;
                let total = self.items_total;
                let device = self.device.clone();
                let tx = tx.clone();

                thread::spawn(move || {
                    let mut offset = idx * batch_size;
                    loop {
                        let length = match offset + batch_size {
                            x if x <= total => batch_size,
                            x if x < total + batch_size => total - offset,
                            _ => 0,
                        };

                        if length > 0 {
                            tx.send(
                                batcher.batch(
                                    lf.clone()
                                        .slice(offset as i64, length as u32)
                                        .collect()
                                        .unwrap(),
                                    transforms.clone(),
                                    &device,
                                ),
                            )
                            .unwrap();
                        } else {
                            break;
                        }

                        offset += num_workers * batch_size;
                    }
                });
            }

            Box::new(InMemoryDataLoaderIterator {
                lf: self.lf.clone(),
                channel: Some(Arc::new(Mutex::new(rx))),
                batcher: self.batcher.clone(),
                transforms: self.transforms.clone(),
                batch_size: self.batch_size,
                items_processed: 0,
                items_total: self.items_total,
                device: &self.device,
            })
        } else {
            Box::new(InMemoryDataLoaderIterator {
                lf: self.lf.clone(),
                channel: None,
                batcher: self.batcher.clone(),
                transforms: self.transforms.clone(),
                batch_size: self.batch_size,
                items_processed: 0,
                items_total: self.items_total,
                device: &self.device,
            })
        }
    }

    fn num_items(&self) -> usize {
        self.items_total
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, Batch<B>>> {
        Arc::new(Self {
            lf: self.lf.clone(),
            batcher: self.batcher.clone(),
            transforms: self.transforms.clone(),
            items_total: self.items_total,
            batch_size: self.batch_size,
            num_workers: self.num_workers,
            device: device.clone(),
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, Batch<B>>> {
        Arc::new(Self {
            lf: self.lf.clone().slice(start as i64, (end - start) as u32),
            batcher: self.batcher.clone(),
            transforms: self.transforms.clone(),
            items_total: self.items_total,
            batch_size: self.batch_size,
            num_workers: self.num_workers,
            device: self.device.clone(),
        })
    }
}

pub struct InMemoryDataLoaderIterator<'a, B: Backend> {
    lf: LazyFrame,
    channel: Option<Arc<Mutex<Receiver<Batch<B>>>>>,
    batcher: Arc<dyn FrameBatcher<B>>,
    transforms: Arc<Pipeline<B>>,
    batch_size: usize,
    items_processed: usize,
    items_total: usize,
    device: &'a B::Device,
}

impl<'a, B: Backend> Iterator for InMemoryDataLoaderIterator<'a, B> {
    type Item = Batch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(recv) = &self.channel {
            self.items_processed += self.batch_size;
            recv.lock().unwrap().recv().ok()
        } else {
            let (offset, length) = match self.items_processed + self.batch_size {
                x if x <= self.items_total => (self.items_processed, self.batch_size),
                x if x < self.items_total + self.batch_size => (
                    self.items_processed,
                    self.items_total - self.items_processed,
                ),
                _ => (0, 0),
            };

            self.items_processed += length;

            (length > 0).then_some(
                self.batcher.batch(
                    self.lf
                        .clone()
                        .slice(offset as i64, length as u32)
                        .collect()
                        .unwrap(),
                    self.transforms.clone(),
                    self.device,
                ),
            )
        }
    }
}

impl<'a, B: Backend> DataLoaderIterator<Batch<B>> for InMemoryDataLoaderIterator<'a, B> {
    fn progress(&self) -> Progress {
        Progress::new(self.items_processed, self.items_total)
    }
}
