use std::marker::PhantomData;

use burn::prelude::Backend;
use polars::prelude::{LazyFrame, PlRefPath, ScanArgsParquet};

use crate::{dataloader::StreamingDataLoader, dataset::StreamableDataset};

pub struct MnistDataset<B: Backend, O> {
    uri: PlRefPath,
    device: B::Device,
    _p: PhantomData<O>,
}

impl<B: Backend, O> MnistDataset<B, O> {
    pub fn new(uri: PlRefPath, device: B::Device) -> Self {
        Self {
            uri,
            device,
            _p: PhantomData,
        }
    }
}

impl<B: Backend, O> StreamableDataset<B, O> for MnistDataset<B, O> {
    fn train(&self, batch_size: usize, shuffle: bool) -> StreamingDataLoader<B, O> {
        let dspath = self.uri.clone().join("**/train-*.parquet");
        let q = LazyFrame::scan_parquet(
            dspath,
            ScanArgsParquet {
                glob: true,
                ..Default::default()
            },
        )
        .unwrap();
        StreamingDataLoader::new(q, batch_size, shuffle, self.device.clone())
    }

    fn val(&self, batch_size: usize, shuffle: bool) -> StreamingDataLoader<B, O> {
        let dspath = self.uri.clone().join("**/test-*.parquet");
        let q = LazyFrame::scan_parquet(
            dspath,
            ScanArgsParquet {
                glob: true,
                ..Default::default()
            },
        )
        .unwrap();
        StreamingDataLoader::new(q, batch_size, shuffle, self.device.clone())
    }

    #[allow(unused_variables)]
    fn test(&self, batch_size: usize, shuffle: bool) -> Option<StreamingDataLoader<B, O>> {
        None
    }
}
