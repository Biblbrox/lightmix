#[cfg(feature = "decode")]
pub mod cifar10;
#[cfg(feature = "decode")]
pub mod cifar100;
#[cfg(feature = "decode")]
pub mod fashionmnist;
#[cfg(feature = "decode")]
pub mod food101;
#[cfg(feature = "decode")]
pub mod imagenet1k;
#[cfg(feature = "decode")]
pub mod mnist;
#[cfg(feature = "decode")]
pub mod tinyimagenet;

#[cfg(feature = "decode")]
use std::io::Cursor;

#[cfg(feature = "decode")]
use image::ImageReader;
use polars::prelude::*;

pub type FrameMapper = Arc<dyn Fn(DataFrame) -> DataFrame + Sync + Send + 'static>;
pub type LazyMapper = Arc<dyn Fn(LazyFrame) -> LazyFrame + Sync + Send + 'static>;

#[cfg(feature = "decode")]
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
