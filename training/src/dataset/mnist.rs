use std::io::Cursor;
use std::sync::Arc;

use burn::data::dataloader::DataLoader;
use burn::prelude::*;
use burn::tensor::DType;

use image::ImageReader;
use polars::prelude::*;

use crate::dataloader::StreamingDataLoader;
use crate::dataset::{PolarsDataset, decode, extract_imagedata, extract_labeldata};

pub struct MnistDataset {
    uri: PlRefPath,
}

impl MnistDataset {
    pub fn new(uri: impl Into<PlRefPath>) -> Self {
        Self { uri: uri.into() }
    }

    pub fn decoder(bytes: &[u8]) -> Vec<u8> {
        let image = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();

        image.into_luma8().to_vec()
    }
}

impl PolarsDataset for MnistDataset {
    fn uri(&self) -> PlRefPath {
        self.uri.clone()
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
        let datasource = LazyFrame::scan_parquet(
            self.uri().join("**/test-*.parquet"),
            ScanArgsParquet {
                glob: true,
                ..Default::default()
            },
        )
        .unwrap();

        Arc::new(StreamingDataLoader::new(
            datasource,
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
        let df = df
            .lazy()
            .with_column(col("image").map(
                |column| decode(column, MnistDataset::decoder),
                |_, _| Ok(Field::new("image".into(), DataType::Binary)),
            ))
            .collect_with_engine(Engine::Streaming)
            .unwrap();

        let imagebuf = extract_imagedata(&df, "image").unwrap();
        let imagedata = TensorData::from_bytes_vec(imagebuf, [batch_size, 1, 28, 28], DType::U8)
            .convert_dtype(DType::F32);
        let images = Tensor::<B, 4>::from_data(imagedata, &device)
            .div_scalar(255)
            .sub_scalar(0.1307)
            .div_scalar(0.3081);

        // Label handling
        let labelbuf = extract_labeldata(&df, "label").unwrap();
        let targets = Tensor::<B, 1, Int>::from_ints(labelbuf.as_slice(), &device);

        Self { images, targets }
    }
}
