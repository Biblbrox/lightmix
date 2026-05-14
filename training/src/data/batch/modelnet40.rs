use std::sync::Arc;

use burn::{
    prelude::Tensor,
    tensor::{DType, Int, TensorData, backend::Backend},
};
use polars::prelude::*;

use crate::data::batch::{CloudBatch, CloudBatcher};

const POINTCOL: &str = "image"; // TODO: For now, I'll use "image", then I'll make the naming better
const LABELCOL: &str = "label";

const NUM_POINTS: usize = 1024;
const NUM_CHANNELS: usize = 3;

pub struct ModelNet40Batcher;

impl ModelNet40Batcher {
    pub fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl<B: Backend> CloudBatcher<B> for ModelNet40Batcher {
    fn batch(&self, df: DataFrame, device: &B::Device) -> CloudBatch<B> {
        let batch_size = df.height();

        // ── Points ────────────────────────────────────────────────────────────
        let total_floats = batch_size * NUM_POINTS * NUM_CHANNELS;
        let mut pointbuf: Vec<u8> = Vec::with_capacity(total_floats * 4);

        df.column(POINTCOL)
            .unwrap()
            .binary()
            .unwrap()
            .into_no_null_iter()
            .for_each(|chunk| pointbuf.extend_from_slice(chunk));

        let pointdata = TensorData::from_bytes_vec(
            pointbuf,
            [batch_size, NUM_POINTS, NUM_CHANNELS],
            DType::F32,
        );

        let points = Tensor::<B, 3>::from_data(pointdata, device);

        // ── Labels ────────────────────────────────────────────────────────────
        let labelbuf: Vec<i32> = df
            .column(LABELCOL)
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();

        let targets = Tensor::<B, 1, Int>::from_ints(labelbuf.as_slice(), device);

        CloudBatch { points, targets }
    }
}
