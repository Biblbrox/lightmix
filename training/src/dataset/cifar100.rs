use std::sync::Arc;

use burn::{data::dataloader::DataLoader, prelude::*, tensor::Int};

use burn::prelude::Backend;
use polars::{
    frame::DataFrame,
    prelude::{LazyFrame, PlRefPath, ScanArgsParquet},
};
use zune_core::bytestream::ZCursor;
use zune_png::PngDecoder;

use crate::dataloader::StreamingDataLoader;
use crate::dataset::PolarsDataset;

#[derive(Clone, Debug)]
pub struct Cifar100Batch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub fine_targets: Tensor<B, 1, Int>,
    pub coarse_targets: Tensor<B, 1, Int>,
}

impl<B: Backend> From<(DataFrame, B::Device)> for Cifar100Batch<B> {
    fn from(value: (DataFrame, B::Device)) -> Cifar100Batch<B> {
        let (df, device) = value;

        let struct_col = df.column("img").unwrap().struct_().unwrap();
        let bytes_col = struct_col.field_by_name("bytes").unwrap();

        let batch_size = bytes_col.len();
        let mut image_buffer: Vec<u8> = Vec::with_capacity(batch_size * 3 * 32 * 32);

        bytes_col.binary().unwrap().iter().for_each(|opt_bytes| {
            let mut decoder = PngDecoder::new(ZCursor::new(opt_bytes.unwrap()));
            let pixels = decoder.decode_raw().unwrap();
            image_buffer.extend(pixels);
        });

        let batch_images = Tensor::<B, 4>::from_data(
            TensorData::new(image_buffer, Shape::new([batch_size, 3, 32, 32]))
                .convert::<B::FloatElem>(),
            &device,
        );

        let batch_size = bytes_col.len();
        let mut coarse_label_buffer: Vec<i64> = Vec::with_capacity(batch_size);
        let mut fine_label_buffer: Vec<i64> = Vec::with_capacity(batch_size);

        df.column("coarse_label")
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .for_each(|label| {
                coarse_label_buffer.extend(vec![label.unwrap()]);
            });
        df.column("fine_label")
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .for_each(|label| {
                fine_label_buffer.extend(vec![label.unwrap()]);
            });

        let batch_coarse_labels = Tensor::<B, 1, Int>::from_data(
            TensorData::new(coarse_label_buffer, Shape::new([batch_size])),
            &device,
        );

        let batch_fine_labels = Tensor::<B, 1, Int>::from_data(
            TensorData::new(fine_label_buffer, Shape::new([batch_size])),
            &device,
        );

        Cifar100Batch {
            images: (batch_images / 255 - 0.1307) / 0.3081,
            fine_targets: batch_fine_labels,
            coarse_targets: batch_coarse_labels,
        }
    }
}

pub struct Cifar100Dataset {
    uri: PlRefPath,
}

impl Cifar100Dataset {
    pub fn new(uri: PlRefPath) -> Self {
        Self { uri }
    }
}

impl PolarsDataset for Cifar100Dataset {
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
