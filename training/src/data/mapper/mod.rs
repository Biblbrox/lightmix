pub mod cifar10;
pub mod cifar100;
pub mod fashionmnist;
pub mod food101;
pub mod imagenet1k;
pub mod mnist;
pub mod tinyimagenet;

use std::io::Cursor;

use image::ImageReader;
use polars::prelude::*;

pub type FrameMapper = Arc<dyn Fn(DataFrame) -> DataFrame + Sync + Send + 'static>;
pub type LazyMapper = Arc<dyn Fn(LazyFrame) -> LazyFrame + Sync + Send + 'static>;

pub fn decode_image_lazy(df: DataFrame, image_col: &'static str) -> DataFrame {
    df.lazy()
        .with_column(col(image_col).map(
            |column| {
                let values: Vec<Vec<u8>> = column
                    .struct_()?
                    .field_by_name("bytes")?
                    .binary()?
                    .into_no_null_iter()
                    .map(|bytes| {
                        let image = ImageReader::new(Cursor::new(bytes))
                            .with_guessed_format()
                            .unwrap()
                            .decode()
                            .unwrap();

                        image.into_luma8().to_vec()
                    })
                    .collect();

                Ok(Column::new(image_col.into(), values))
            },
            |_, _| Ok(Field::new(image_col.into(), DataType::Binary)),
        ))
        .collect_with_engine(Engine::Streaming)
        .unwrap()
}
