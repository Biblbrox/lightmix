use burn::{
    backend::Autodiff,
    module::{AutodiffModule, Module},
    prelude::Backend,
    train::{ClassificationOutput, InferenceStep, TrainStep},
};

use crate::data::batch::ImageBatch;

pub mod fast_vit;
//pub mod swin;
pub mod parallel_vit;
pub mod self_vit;
pub mod token_to_token;
pub mod vit;

pub trait ModelConfig<B: Backend> {
    type ValidModel: Module<B>
        + InferenceStep<Input = ImageBatch<B>, Output = ClassificationOutput<B>>;
    type TrainModel: AutodiffModule<Autodiff<B>, InnerModule = Self::ValidModel>
        + TrainStep<Input = ImageBatch<Autodiff<B>>, Output = ClassificationOutput<Autodiff<B>>>
        + core::fmt::Display
        + 'static;

    fn init_training(&self, device: &B::Device) -> Self::TrainModel;
    fn init_inference(&self, device: &B::Device) -> Self::ValidModel;
}
