use std::io::Cursor;

use image::ImageReader;
use polars::prelude::*;

use crate::data::mapper::FrameMapper;

const IMAGECOL: &str = "img";

pub struct Cifar10Mapper;

impl Cifar10Mapper {
    pub fn decoder() -> FrameMapper {
        Arc::new(|df| {
            df.lazy()
                .with_column(col(IMAGECOL).map(
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

                                image.into_rgb8().to_vec()
                            })
                            .collect();

                        Ok(Column::new(IMAGECOL.into(), values))
                    },
                    |_, _| Ok(Field::new(IMAGECOL.into(), DataType::Binary)),
                ))
                .collect_with_engine(Engine::Streaming)
                .unwrap()
        })
    }
}
