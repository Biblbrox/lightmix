use std::io::Cursor;

use burn::{prelude::*, tensor::DType};
use image::{ImageReader, imageops::FilterType};
use polars::prelude::*;

use crate::dataset::{PolarsDataset, decode, extract_imagedata, extract_labeldata};

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
                |column| decode(column, ImageNet1kDataset::decoder),
                |_, _| Ok(Field::new("image".into(), DataType::Binary)),
            ))
            .collect_with_engine(Engine::Streaming)
            .unwrap();

        let imagebuf = extract_imagedata(&df, "image").unwrap();
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
        let labelbuf = extract_labeldata(&df, "label").unwrap();
        let targets = Tensor::<B, 1, Int>::from_ints(labelbuf.as_slice(), &device);

        Self { images, targets }
    }
}

pub struct ImageNet1kDataset {
    uri: PlRefPath,
}

impl ImageNet1kDataset {
    pub fn new(uri: PlRefPath) -> Self {
        Self { uri }
    }

    pub fn decoder(bytes: &[u8]) -> Vec<u8> {
        let image = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap()
            .resize_exact(256, 256, FilterType::Triangle)
            .crop_imm(16, 16, 224, 224);

        image.to_rgb8().to_vec()
    }
}

impl PolarsDataset for ImageNet1kDataset {
    fn uri(&self) -> PlRefPath {
        self.uri.clone()
    }
}
