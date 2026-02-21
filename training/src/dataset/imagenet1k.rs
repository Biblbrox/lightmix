use std::sync::Arc;

use burn::{
    Tensor,
    data::dataloader::DataLoader,
    prelude::{Backend, ToElement},
    tensor::{Int, Shape, TensorData},
};
use polars::{
    frame::DataFrame,
    prelude::{
        Column, DataType, Engine, Field, IntoLazy, LazyFrame, PlRefPath, ScanArgsParquet, col,
    },
};
use zune_image::{
    codecs::{jpeg::JpegDecoder, qoi::zune_core::bytestream::ZCursor},
    image::Image,
    traits::OperationsTrait,
};
use zune_imageprocs::{
    crop::Crop,
    resize::{Resize, ResizeMethod},
};
use zune_core::{bytestream::ZCursor, options::DecoderOptions};
use zune_image::{codecs::jpeg::JpegDecoder, image::Image, traits::OperationsTrait};
use zune_imageprocs::resize::Resize;
use zune_png::PngDecoder;

use crate::{dataloader::StreamingDataLoader, dataset::PolarsDataset};

pub struct ImageNet1kDataset {
    uri: PlRefPath,
}

#[derive(Clone, Debug)]
pub struct ImageNet1kBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> From<(DataFrame, B::Device)> for ImageNet1kBatch<B> {
    fn from(value: (DataFrame, B::Device)) -> ImageNet1kBatch<B> {
        let (df, device) = value;

        let struct_col = df.column("image").unwrap().struct_().unwrap();
        let bytes_col = struct_col.field_by_name("bytes").unwrap();

        let batch_size = bytes_col.len();
        let mut image_buffer: Vec<u8> = Vec::with_capacity(batch_size * 3 * 224 * 224);

        bytes_col.binary().unwrap().iter().for_each(|opt_bytes| {
            let options = DecoderOptions::default();
            let mut image = Image::read(ZCursor::new(opt_bytes.unwrap()), options).unwrap();

            Resize::new(224, 224, zune_imageprocs::resize::ResizeMethod::Bilinear)
                .execute(&mut image)
                .unwrap();
            image_buffer.extend(image.flatten_to_u8().iter().flatten());
        });

        let batch_images = Tensor::<B, 4>::from_data(
            TensorData::new(image_buffer, Shape::new([batch_size, 3, 224, 224]))
                .convert::<B::FloatElem>(),
            &device,
        );

        let batch_size = bytes_col.len();
        let mut label_buffer: Vec<i64> = Vec::with_capacity(batch_size);

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

        ImageNet1kBatch {
            images: (batch_images / 255 - 0.1307) / 0.3081,
            targets: batch_labels,
        }
    }
}

impl ImageNet1kDataset {
    pub fn new(uri: PlRefPath) -> Self {
        Self { uri }
    }
}

impl PolarsDataset for ImageNet1kDataset {
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
        let dspath = self.uri.clone().join("**/validation-*.parquet");
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

    fn test<B, O>(
        &self,
        batch_size: usize,
        shuffle_seed: Option<u64>,
        device: &B::Device,
    ) -> Option<Arc<dyn DataLoader<B, O>>>
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
        Some(Arc::new(StreamingDataLoader::new(
            q,
            batch_size,
            shuffle_seed,
            device,
        )))
    }
}
