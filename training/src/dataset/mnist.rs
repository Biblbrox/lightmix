use std::{marker::PhantomData, sync::Arc};

use burn::{data::dataloader::DataLoader, prelude::*, tensor::Int};

use polars::prelude::{LazyFrame, PlRefPath, ScanArgsParquet};

use crate::{dataloader::StreamingDataLoader, dataset::StreamableDataset};

use polars::frame::DataFrame;
use zune_core::bytestream::ZCursor;
use zune_png::PngDecoder;

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

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> From<DataFrame> for MnistBatch<B> {
    fn from(value: DataFrame) -> MnistBatch<B> {
        let device = B::Device::default();

        let struct_col = value.column("image").unwrap().struct_().unwrap();
        let bytes_col = struct_col.field_by_name("bytes").unwrap();

        let batch_size = bytes_col.len();
        let mut image_buffer: Vec<u8> = Vec::with_capacity(batch_size * 28 * 28);

        bytes_col.binary().unwrap().iter().for_each(|opt_bytes| {
            let mut decoder = PngDecoder::new(ZCursor::new(opt_bytes.unwrap()));
            let pixels = decoder.decode_raw().unwrap();
            image_buffer.extend(pixels);
        });

        let batch_images = Tensor::<B, 4>::from_data(
            TensorData::new(image_buffer, Shape::new([batch_size, 1, 28, 28]))
                .convert::<B::FloatElem>(),
            &device,
        );

        let batch_size = bytes_col.len();
        let mut label_buffer: Vec<i64> = Vec::with_capacity(batch_size * 28 * 28);

        value
            .column("label")
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .for_each(|label| {
                label_buffer.extend(vec![label.unwrap()]);
            });

        let batch_labels = Tensor::<B, 1, Int>::from_data(
            TensorData::new(label_buffer, Shape::new([batch_size])),
            &device,
        );

        MnistBatch {
            images: (batch_images / 255 - 0.1307) / 0.3081,
            targets: batch_labels,
        }
    }
}

impl<B, O> StreamableDataset<B, O> for MnistDataset<B, O>
where
    B: Backend,
    O: std::convert::From<polars::prelude::DataFrame> + Clone + Send + Sync + 'static,
{
    fn train(&self, batch_size: usize, shuffle: bool) -> Arc<dyn DataLoader<B, O>> {
        let dspath = self.uri.clone().join("**/train-*.parquet");
        let q = LazyFrame::scan_parquet(
            dspath,
            ScanArgsParquet {
                glob: true,
                ..Default::default()
            },
        )
        .unwrap();
        Arc::new(StreamingDataLoader::new(
            q,
            batch_size,
            shuffle,
            self.device.clone(),
        ))
    }

    fn val(&self, batch_size: usize, shuffle: bool) -> Arc<dyn DataLoader<B, O>> {
        let dspath = self.uri.clone().join("**/test-*.parquet");
        let q = LazyFrame::scan_parquet(
            dspath,
            ScanArgsParquet {
                glob: true,
                ..Default::default()
            },
        )
        .unwrap();
        Arc::new(StreamingDataLoader::new(
            q,
            batch_size,
            shuffle,
            self.device.clone(),
        ))
    }

    #[allow(unused_variables)]
    fn test(&self, batch_size: usize, shuffle: bool) -> Option<Arc<dyn DataLoader<B, O>>> {
        None
    }
}
