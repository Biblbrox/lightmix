use std::sync::Arc;

use burn::tensor::DType;
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
    fn from(value: (DataFrame, B::Device)) -> Self {
        let (df, device) = value;
        let batch_size = df.height();

        // Image handling
        let mut imagebuf = vec![0; batch_size * 28 * 28];
        for (idx, bytes) in df
            .column("image")
            .unwrap()
            .struct_()
            .unwrap()
            .field_by_name("bytes")
            .unwrap()
            .binary()
            .unwrap()
            .into_no_null_iter()
            .enumerate()
        {
            let mut decoder = PngDecoder::new(ZCursor::new(bytes));
            let slice = &mut imagebuf[idx * 28 * 28..(idx + 1) * 28 * 28];
            decoder.decode_into(slice).unwrap();
        }

        let imagedata = TensorData::from_bytes_vec(imagebuf, [batch_size, 1, 28, 28], DType::U8)
            .convert_dtype(DType::F64);
        let images = Tensor::<B, 4>::from_data(imagedata, &device)
            .div_scalar(255)
            .sub_scalar(0.1307)
            .div_scalar(0.3081);

        // Label handling
        let labelbuf: Vec<i64> = df
            .column("label")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let targets = Tensor::<B, 1, Int>::from_ints(labelbuf.as_slice(), &device);

        Self { images, targets }
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
