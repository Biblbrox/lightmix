use polars::prelude::*;

pub mod buffered;
pub mod fixed;

pub trait FrameBatchStrategy: Send + Sync {
    fn init(&mut self, stream: CollectBatches);
    fn batch(&mut self) -> Option<DataFrame>;
    fn batch_size(&self) -> usize;
    fn chunk_size(&self) -> usize;
    fn clone_dyn(&self) -> Box<dyn FrameBatchStrategy>;
}
