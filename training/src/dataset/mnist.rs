use std::sync::Arc;

use burn::tensor::DType;
use burn::{data::dataloader::DataLoader, prelude::*, tensor::Int};

use polars::frame::DataFrame;
use polars::prelude::{
    Column, DataType, Engine, Field, IntoLazy, LazyFrame, PlRefPath, ScanArgsParquet, col,
};
use zune_image::codecs::png::PngDecoder;
use zune_image::codecs::qoi::zune_core::bytestream::ZCursor;

use crate::dataloader::StreamingDataLoader;
use crate::dataset::PolarsDataset;

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
        // Parallel PNG decoding using LazyAPI
        let df = df
            .lazy()
            .with_column(col("image").map(
                |col| {
                    Ok(Column::new::<Vec<Vec<u8>>, _>(
                        "image".into(),
                        col.struct_()
                            .unwrap()
                            .field_by_name("bytes")
                            .unwrap()
                            .binary()
                            .unwrap()
                            .into_no_null_iter()
                            .map(|bytes| {
                                let mut decoder = PngDecoder::new(ZCursor::new(bytes));
                                decoder.decode_raw().unwrap()
                            })
                            .collect(),
                    ))
                },
                |_, _| Ok(Field::new("image".into(), DataType::Binary)),
            ))
            .collect_with_engine(Engine::Streaming)
            .unwrap();

        // Flat image bytes extraction
        let imagebuf = df
            .column("image")
            .unwrap()
            .binary()
            .unwrap()
            .into_no_null_iter()
            .flatten()
            .copied()
            .collect();

        let imagedata = TensorData::from_bytes_vec(imagebuf, [batch_size, 1, 28, 28], DType::U8)
            .convert_dtype(DType::F32);
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
