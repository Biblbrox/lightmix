use std::{marker::PhantomData, num::NonZero, sync::Arc};

use burn::{
    data::dataloader::{DataLoader, DataLoaderIterator, Progress},
    prelude::Backend,
};
use polars::{
    frame::DataFrame,
    prelude::{CollectBatches, Engine, LazyFrame, len},
};

pub struct StreamingDataLoader<B: Backend, O> {
    dataquery: LazyFrame,
    batch_size: usize,
    shuffle_seed: Option<u64>,
    total_items: usize,
    device: B::Device,
    _o: PhantomData<O>,
}

impl<B: Backend, O> StreamingDataLoader<B, O> {
    pub fn new(
        datasource: impl Into<LazyFrame>,
        batch_size: usize,
        shuffle_seed: Option<u64>,
        device: &B::Device,
    ) -> Self {
        let dataquery = datasource.into();

        // LazyFrame doesn't have a `height` method to count rows, so we run `select[len()]` query
        // once here and cache the resulting cell value. It should be a fast metadata query when
        // run against parquet files (see https://github.com/pola-rs/polars/issues/11404)
        let total_items = dataquery
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
            dataquery,
            batch_size,
            shuffle_seed,
            total_items,
            device: device.clone(),
            _o: PhantomData,
        }
    }
}

impl<B, O> DataLoader<B, O> for StreamingDataLoader<B, O>
where
    B: Backend,
    O: From<(DataFrame, B::Device)> + Sync + Send + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        Box::new(StreamingDataLoaderIterator::<B, O>::new(
            self.dataquery.clone(),
            self.batch_size,
            self.shuffle_seed,
            self.total_items,
            &self.device,
        ))
    }

    fn num_items(&self) -> usize {
        self.total_items
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, O>> {
        Arc::new(StreamingDataLoader::new(
            self.dataquery.clone(),
            self.batch_size,
            self.shuffle_seed,
            device,
        ))
    }

    // NOTE: Slice will be truncated if end is outside the row count
    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, O>> {
        Arc::new(StreamingDataLoader::new(
            self.dataquery
                .clone()
                .slice(start as i64, (end - start) as u32),
            self.batch_size,
            self.shuffle_seed,
            &self.device,
        ))
    }
}

pub struct StreamingDataLoaderIterator<B: Backend, O> {
    batches: CollectBatches,
    batch_size: usize,
    current_batch: usize,
    total_items: usize,
    shuffle_seed: Option<u64>,
    device: B::Device,
    _p: PhantomData<O>,
}

impl<B: Backend, O> StreamingDataLoaderIterator<B, O> {
    pub fn new(
        datasource: LazyFrame,
        batch_size: usize,
        shuffle_seed: Option<u64>,
        total_items: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            batches: datasource
                .collect_batches(
                    Engine::Auto,
                    shuffle_seed.is_none(),
                    NonZero::new(batch_size),
                    true,
                )
                .unwrap(),
            batch_size,
            current_batch: 0,
            total_items,
            shuffle_seed,
            device: device.clone(),
            _p: PhantomData,
        }
    }
}

impl<B: Backend, O: From<(DataFrame, B::Device)>> Iterator for StreamingDataLoaderIterator<B, O> {
    type Item = O;

    fn next(&mut self) -> Option<Self::Item> {
        let device = self.device.clone();
        self.current_batch += 1;
        self.batches.next().transpose().ok().flatten().map(|df| {
            // Judging by the Polars source code, resampling full dataset without shuffling is
            // potentially a no-op, so we don't need to force branching on shuffle_seed here
            let df = df
                .sample_n_literal(
                    df.height(),
                    false,
                    self.shuffle_seed.is_some(),
                    self.shuffle_seed,
                )
                .unwrap();
            O::from((df, device))
        })
    }
}

impl<B: Backend, O: From<(DataFrame, B::Device)>> DataLoaderIterator<O>
    for StreamingDataLoaderIterator<B, O>
{
    // Not sure if current_batch x batch_size may end up bigger than total_items, so watch out
    // for any weird execution failures
    fn progress(&self) -> Progress {
        Progress::new(self.current_batch * self.batch_size, self.total_items)
    }
}
