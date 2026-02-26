use std::io::Cursor;
use std::sync::Arc;

use burn::data::dataloader::DataLoader;
use burn::prelude::*;

use burn::tensor::DType;
use image::ImageReader;
use polars::prelude::*;

use crate::dataloader::StreamingDataLoader;
use crate::dataset::{PolarsDataset, decode, extract_imagedata, extract_labeldata};

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
        let df = df
            .lazy()
            .with_column(col("img").map(
                |column| decode(column, Cifar100Dataset::decoder),
                |_, _| Ok(Field::new("img".into(), DataType::Binary)),
            ))
            .collect_with_engine(Engine::Streaming)
            .unwrap();

        let imagebuf = extract_imagedata(&df, "img").unwrap();
        let imagedata = TensorData::from_bytes_vec(imagebuf, [batch_size, 32, 32, 3], DType::U8)
            .convert_dtype(DType::F32);

        let mean = Tensor::<B, 1>::from_floats([0.5071, 0.4867, 0.4408], &device)
            .reshape([1, 3, 1, 1])
            .expand([batch_size, 3, 32, 32]);

        let std = Tensor::<B, 1>::from_floats([0.2675, 0.2565, 0.2761], &device)
            .reshape([1, 3, 1, 1])
            .expand([batch_size, 3, 32, 32]);

        let images = Tensor::<B, 4>::from_data(imagedata, &device)
            .swap_dims(1, -1)
            .div_scalar(255)
            .sub(mean)
            .div(std);

        // Label handling
        let fine_labelbuf = extract_labeldata(&df, "fine_label").unwrap();
        let fine_targets = Tensor::<B, 1, Int>::from_ints(fine_labelbuf.as_slice(), &device);

        let coarse_labelbuf = extract_labeldata(&df, "coarse_label").unwrap();
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

    pub fn decoder(bytes: &[u8]) -> Vec<u8> {
        let image = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();

        image.into_rgb8().to_vec()
    }
}

impl PolarsDataset for Cifar100Dataset {
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
        let dspath = self.uri().join("**/test-*.parquet");
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
