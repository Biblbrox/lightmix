use std::sync::Arc;

use burn::{data::dataloader::DataLoader, prelude::*, tensor::Int};

use polars::frame::DataFrame;
use polars::prelude::{LazyFrame, PlRefPath, ScanArgsParquet};

use crate::dataloader::StreamingDataLoader;
use crate::dataset::PolarsDataset;

use zune_core::bytestream::ZCursor;
use zune_png::PngDecoder;

pub struct MnistDataset {
    uri: PlRefPath,
}

impl MnistDataset {
    pub fn new(uri: PlRefPath) -> Self {
        Self { uri }
    }
}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> From<(DataFrame, B::Device)> for MnistBatch<B> {
    fn from(value: (DataFrame, B::Device)) -> MnistBatch<B> {
        let (df, device) = value;

        let struct_col = df.column("image").unwrap().struct_().unwrap();
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

        df.column("label")
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

impl PolarsDataset for MnistDataset {
    fn train<B, O>(
        &self,
        batch_size: usize,
        shuffle_seed: Option<u64>,
        device: &B::Device,
    ) -> Arc<dyn DataLoader<B, O>>
    where
        B: Backend,
        O: From<(DataFrame, B::Device)> + Sync + Send + 'static,
    {
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
            shuffle_seed,
            device,
        ))
    }

    fn val<B, O>(
        &self,
        batch_size: usize,
        shuffle_seed: Option<u64>,
        device: &B::Device,
    ) -> Arc<dyn DataLoader<B, O>>
    where
        B: Backend,
        O: From<(DataFrame, B::Device)> + Sync + Send + 'static,
    {
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
            shuffle_seed,
            device,
        ))
    }

    #[allow(unused_variables)]
    fn test<B: Backend, O>(
        &self,
        batch_size: usize,
        shuffle_seed: Option<u64>,
        device: &B::Device,
    ) -> Option<Arc<dyn DataLoader<B, O>>> {
        None
    }
}
