use std::sync::Arc;

use burn::{
    Tensor,
    data::dataloader::DataLoader,
    module::Module,
    prelude::Backend,
    tensor::{DType, Int, TensorData},
};
use polars::{
    frame::DataFrame,
    prelude::{
        Column, DataType, Engine, Field, IntoLazy, LazyFrame, PlRefPath, ScanArgsParquet, col,
    },
};
use zune_core::options::DecoderOptions;
use zune_image::{
    codecs::{jpeg::JpegDecoder, png::PngDecoder, qoi::zune_core::bytestream::ZCursor},
    image::Image,
    traits::OperationsTrait,
};
use zune_imageprocs::{
    crop::Crop,
    resize::{Resize, ResizeMethod},
};

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
    fn from(value: (DataFrame, B::Device)) -> Self {
        let (df, device) = value;
        let batch_size = df.height();

        // Parallel JPEG decoding and resizing using LazyAPI
        let df = df
            .lazy()
            .with_column(col("image").map(
                |col| {
                    let resizer = Resize::new(256, 256, ResizeMethod::Bicubic);
                    let cropper = Crop::new(224, 224, 16, 16);
                    Ok(Column::new::<Vec<Vec<u8>>, _>(
                        "image".into(),
                        col.struct_()
                            .unwrap()
                            .field_by_name("bytes")
                            .unwrap()
                            .binary()
                            .unwrap()
                            .into_no_null_iter()
                            .map(|bytes| -> Vec<u8> {
                                // Decode JPEG into Image
                                let mut image = match Image::from_decoder(JpegDecoder::new(
                                    ZCursor::new(bytes),
                                )) {
                                    Ok(img) => img,
                                    Err(_) => {
                                        Image::from_decoder(PngDecoder::new(ZCursor::new(bytes)))
                                            .unwrap()
                                    }
                                };

                                // Resize to 256x256
                                resizer.execute(&mut image).unwrap();

                                // Center-crop to 224x224
                                cropper.execute(&mut image).unwrap();

                                // Flatten to raw RGB bytes
                                image.flatten_frames().concat()
                            })
                            .collect(),
                    ))
                },
                |_, _| Ok(Field::new("image".into(), DataType::Binary)),
            ))
            .collect_with_engine(Engine::Streaming)
            .unwrap();

        let imagebuf = df
            .column("image")
            .unwrap()
            .binary()
            .unwrap()
            .into_no_null_iter()
            .flatten()
            .copied()
            .collect();

        let imagedata = TensorData::from_bytes_vec(imagebuf, [batch_size, 224, 224, 3], DType::U8)
            .convert_dtype(DType::F32);

        // Tensor packing with normalization
        let mean = Tensor::<B, 1>::from_floats([0.485, 0.456, 0.406], &device)
            .reshape([1, 3, 1, 1])
            .expand([batch_size, 3, 224, 224]);

        let std = Tensor::<B, 1>::from_floats([0.229, 0.224, 0.225], &device)
            .reshape([1, 3, 1, 1])
            .expand([batch_size, 3, 224, 224]);

        let images = Tensor::<B, 4>::from_data(imagedata, &device)
            .swap_dims(1, -1)
            .div_scalar(255)
            .sub(mean)
            .div(std);

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
