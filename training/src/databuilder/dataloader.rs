use std::{marker::PhantomData, num::NonZero, sync::Arc};

use burn::{
    data::dataloader::{DataLoader, DataLoaderIterator, Progress},
    prelude::Backend,
};
use polars::{
    frame::DataFrame,
    prelude::{CollectBatches, Engine, LazyFrame, len},
};

/// This dataloader holds a LazyFrame of a data query to produce a stream of batches for the consumer.
/// It is generic over input type I which should implement Into<LazyFrame> and over output type O
/// which should implement From<DataFrame>
pub struct StreamingDataLoader<B: Backend, I, O> {
    dataquery: LazyFrame,
    batch_size: usize,
    shuffle: bool,
    total_items: usize,
    device: B::Device,
    _i: PhantomData<I>,
    _o: PhantomData<O>,
}

impl<B: Backend, I: Into<LazyFrame>, O> StreamingDataLoader<B, I, O> {
    /// Constructs a data query LazyFrame from the input source using its Into<LazyFrame> trait.
    ///
    /// # Notes
    ///
    /// * `shuffle` - is a reciprocal of CollectBatches' `maintain_order` and does not actually
    ///   guarantee a proper shuffle.
    ///
    /// * `total_items` - is calculated here once by eagerly collecting a `len` agg over a source
    ///   LazyFrame; it *should* be a fast operation if sources are parquet or arrow files, as they
    ///   contain number of rows in their metadata (see https://github.com/pola-rs/polars/issues/11404)
    pub fn new(datasource: I, batch_size: usize, shuffle: bool, device: B::Device) -> Self {
        let dataquery = datasource.into();
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
            shuffle,
            total_items,
            device,
            _i: PhantomData,
            _o: PhantomData,
        }
    }
}

impl<B, I, O> DataLoader<B, O> for StreamingDataLoader<B, I, O>
where
    B: Backend,
    I: Sync + Send,
    O: From<DataFrame> + Sync + Send + 'static,
{
    /// Creates a StreamingDataLoaderIterator instance
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        Box::new(StreamingDataLoaderIterator::new(
            self.dataquery.clone(),
            self.batch_size,
            self.shuffle,
            self.total_items,
        ))
    }

    fn num_items(&self) -> usize {
        self.total_items
    }

    /// This actually doesn't do much, just creates a copy with changed `device` value
    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, O>> {
        Arc::new(StreamingDataLoader::new(
            self.dataquery.clone(),
            self.batch_size,
            self.shuffle,
            device.clone(),
        ))
    }

    /// This creates a copy which adds a `slice` operation to the query graph. Burn`s `start` and `end`
    /// translate to Polars` `offset` and `len`. Slice will be truncated if offset+len is bigger
    /// than row count
    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, O>> {
        Arc::new(StreamingDataLoader::new(
            self.dataquery
                .clone()
                .slice(start as i64, (end - start) as u32),
            self.batch_size,
            self.shuffle,
            self.device.clone(),
        ))
    }
}

/// Basically a tracking iterator over DataFrame batches with a method to calculate progress
pub struct StreamingDataLoaderIterator<O> {
    batches: CollectBatches,
    batch_size: usize,
    current_batch: usize,
    total_items: usize,
    phantom: PhantomData<O>,
}

impl<O> StreamingDataLoaderIterator<O> {
    pub fn new(
        datasource: LazyFrame,
        batch_size: usize,
        shuffle: bool,
        total_items: usize,
    ) -> Self {
        Self {
            batches: datasource
                .collect_batches(Engine::Auto, !shuffle, NonZero::new(batch_size), true)
                .unwrap(),
            batch_size,
            current_batch: 0,
            total_items,
            phantom: PhantomData,
        }
    }
}

impl<O: From<DataFrame>> Iterator for StreamingDataLoaderIterator<O> {
    type Item = O;

    /// Next batch actually returns an Option<Result<DataFrame>>, so we transpose it, turn Result into
    /// Option, and flatten double Options to a single Option<DataFrame>, which is then mapped to
    /// the output O type using its From<DataFrame> trait
    fn next(&mut self) -> Option<Self::Item> {
        self.current_batch += 1;
        self.batches
            .next()
            .transpose()
            .ok()
            .flatten()
            .map(|df| O::from(df))
    }
}

impl<O: From<DataFrame>> DataLoaderIterator<O> for StreamingDataLoaderIterator<O> {
    /// I'm not sure if current_batch x batch_size may end up bigger than total_items, so watch out
    /// for any weird execution failures
    fn progress(&self) -> Progress {
        Progress::new(self.current_batch * self.batch_size, self.total_items)
    }
}
