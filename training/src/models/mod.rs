use burn::{
    backend::Autodiff,
    module::{AutodiffModule, Module},
    tensor::backend::Backend,
    train::{ClassificationOutput, InferenceStep, TrainStep},
};

use crate::data::batch::ImageBatch;

pub mod efficientvit;
pub mod fast_vit;
pub mod token_to_token;
pub mod vit;

pub trait ModelConfig<B: Backend> {
    type ValidModel: Module<B>
        + InferenceStep<Input = ImageBatch<B>, Output = ClassificationOutput<B>>;
    type TrainModel: AutodiffModule<Autodiff<B>, InnerModule = Self::ValidModel>
        + TrainStep<Input = ImageBatch<Autodiff<B>>, Output = ClassificationOutput<Autodiff<B>>>
        + core::fmt::Display
        + 'static;

    fn init_training(
        &self,
        device: &B::Device,
        in_channels: usize,
        image_size: usize,
        num_classes: usize,
    ) -> Self::TrainModel;
    fn init_inference(
        &self,
        device: &B::Device,
        in_channels: usize,
        image_size: usize,
        num_classes: usize,
    ) -> Self::ValidModel;
}
