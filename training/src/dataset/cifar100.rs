use std::sync::Arc;

use burn::tensor::DType;
use burn::{data::dataloader::DataLoader, prelude::*, tensor::Int};

use burn::prelude::Backend;
use polars::prelude::{Column, DataType, Engine, Field, IntoLazy, col};
use polars::{
    frame::DataFrame,
    prelude::{LazyFrame, PlRefPath, ScanArgsParquet},
};
use zune_image::codecs::png::PngDecoder;
use zune_image::codecs::qoi::zune_core::bytestream::ZCursor;

use crate::dataloader::StreamingDataLoader;
use crate::dataset::PolarsDataset;

#[derive(Clone, Debug)]
pub struct Cifar100Batch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub fine_targets: Tensor<B, 1, Int>,
    pub coarse_targets: Tensor<B, 1, Int>,
}

impl<B: Backend> From<(DataFrame, B::Device)> for Cifar100Batch<B> {
    fn from(value: (DataFrame, B::Device)) -> Self {
        let (df, device) = value;
        let batch_size = df.height();

        // Image handling
        // Parallel PNG decoding using LazyAPI
        let df = df
            .lazy()
            .with_column(col("img").map(
                |col| {
                    Ok(Column::new::<Vec<Vec<u8>>, _>(
                        "img".into(),
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
                |_, _| Ok(Field::new("img".into(), DataType::Binary)),
            ))
            .collect_with_engine(Engine::Streaming)
            .unwrap();

        // Flat image bytes extraction
        let imagebuf = df
            .column("img")
            .unwrap()
            .binary()
            .unwrap()
            .into_no_null_iter()
            .flatten()
            .copied()
            .collect();

        // Tensor packing with normalization
        let rmean = Tensor::<B, 3>::full([batch_size, 32, 32], 0.5071, &device);
        let gmean = Tensor::<B, 3>::full([batch_size, 32, 32], 0.4867, &device);
        let bmean = Tensor::<B, 3>::full([batch_size, 32, 32], 0.4408, &device);
        let mean = Tensor::stack::<4>(vec![rmean, gmean, bmean], 1);

        let rstd = Tensor::<B, 3>::full([batch_size, 32, 32], 0.2675, &device);
        let gstd = Tensor::<B, 3>::full([batch_size, 32, 32], 0.2565, &device);
        let bstd = Tensor::<B, 3>::full([batch_size, 32, 32], 0.2761, &device);
        let std = Tensor::stack::<4>(vec![rstd, gstd, bstd], 1);

        let imagedata = TensorData::from_bytes_vec(imagebuf, [batch_size, 32, 32, 3], DType::U8)
            .convert_dtype(DType::F32);
        let images = Tensor::<B, 4>::from_data(imagedata, &device)
            .swap_dims(1, -1)
            .div_scalar(255)
            .sub(mean)
            .div(std);

        // Label handling
        let fine_labelbuf: Vec<i64> = df
            .column("fine_label")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let fine_targets = Tensor::<B, 1, Int>::from_ints(fine_labelbuf.as_slice(), &device);

        let coarse_labelbuf: Vec<i64> = df
            .column("coarse_label")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let coarse_targets = Tensor::<B, 1, Int>::from_ints(coarse_labelbuf.as_slice(), &device);

        Self {
            images,
            fine_targets,
            coarse_targets,
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
