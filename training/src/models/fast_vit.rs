use burn::{
    Tensor,
    backend::Autodiff,
    module::Module,
    nn::{Linear, LinearConfig, loss::CrossEntropyLossConfig},
    tensor::{
        Int,
        backend::{AutodiffBackend, Backend},
    },
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use serde::Deserialize;

use crate::{
    data::batch::Batch,
    embeddings::vit::{PatchEmbedding, PatchEmbeddingConfig},
    encoders::fast_encoder::{FastEncoder, FastEncoderConfig},
    models::ModelConfig,
    norm::{DynamicERF, DynamicERFConfig},
};

#[derive(Module, Debug)]
pub struct FastViT<B: Backend> {
    embedding_block: PatchEmbedding<B>,
    encoder: FastEncoder<B>,
    layer_norm: DynamicERF<B>,
    linear: Linear<B>,
    in_channels: usize,
    image_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FastViTConfig {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_encoders: usize,
    pub patch_size: usize,
    pub hidden_dim: usize,
    pub dropout: f64,
    pub sinkhorn_temp: f32,
    pub activation: String,
}

impl<B: Backend> FastViT<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.embedding_block.forward(images);
        let x = self.encoder.forward(x);
        let x = self.layer_norm.forward(x);

        self.linear.forward(x.mean_dim(1)).squeeze()
    }
}

impl FastViTConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
        in_channels: usize,
        image_size: usize,
        num_classes: usize,
    ) -> FastViT<B> {
        let grid_size = image_size / self.patch_size;
        let num_patches = grid_size.pow(2);

        FastViT {
            embedding_block: PatchEmbeddingConfig::new(
                in_channels,
                self.embed_dim,
                self.patch_size,
                image_size,
                self.dropout,
                num_patches,
                false,
            )
            .init(device),

            encoder: FastEncoderConfig::new(
                self.num_encoders,
                grid_size,
                num_patches,
                self.embed_dim,
                self.num_heads,
                self.hidden_dim,
                self.dropout,
                self.sinkhorn_temp,
            )
            .init(device),
            layer_norm: DynamicERFConfig::new(self.embed_dim).init(device),
            linear: LinearConfig::new(self.embed_dim, num_classes).init(device),
            in_channels,
            image_size,
        }
    }

    pub fn model_name(&self) -> String {
        format!(
            "fast_vit-head{}-hid{}-emb{}-enc{}-temp{}",
            self.num_heads, self.hidden_dim, self.embed_dim, self.num_encoders, self.sinkhorn_temp
        )
    }
}

impl<B: Backend> ModelConfig<B> for FastViTConfig {
    type TrainModel = FastViT<Autodiff<B>>;
    type ValidModel = FastViT<B>;

    fn init_training(
        &self,
        device: &B::Device,
        in_channels: usize,
        image_size: usize,
        num_classes: usize,
    ) -> Self::TrainModel {
        self.init(device, in_channels, image_size, num_classes)
    }

    fn init_inference(
        &self,
        device: &B::Device,
        in_channels: usize,
        image_size: usize,
        num_classes: usize,
    ) -> Self::ValidModel {
        self.init(device, in_channels, image_size, num_classes)
    }
}

impl<B: Backend> FastViT<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep for FastViT<B> {
    type Input = Batch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Batch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let images = batch.data.clone().reshape([
            batch.batch_size(),
            self.in_channels,
            self.image_size,
            self.image_size,
        ]);
        let item = self.forward_classification(images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for FastViT<B> {
    type Input = Batch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Batch<B>) -> ClassificationOutput<B> {
        let images = batch.data.clone().reshape([
            batch.batch_size(),
            self.in_channels,
            self.image_size,
            self.image_size,
        ]);
        self.forward_classification(images, batch.targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::fast_vit::FastViTConfig;
    use burn::{
        backend::{Flex, flex::FlexDevice},
        tensor::Shape,
    };

    type B = Flex;
    type Device = FlexDevice;

    const IN_CHANNELS: usize = 3;
    const PATCH_SIZE: usize = 4;
    const IMG_SIZE: usize = 32;
    const EMBED_DIM: usize = PATCH_SIZE.pow(2) * IN_CHANNELS;
    const NUM_HEADS: usize = 8;
    const NUM_ENCODERS: usize = 4;
    const NUM_CLASSES: usize = 10;
    const BATCH_SIZE: usize = 10;
    const HIDDEN_DIM: usize = 64;
    const DROPOUT: f64 = 0.1;
    const SINKHORN_TEMP: f64 = 0.05;

    fn test_config() -> FastViTConfig {
        FastViTConfig {
            embed_dim: EMBED_DIM,
            num_heads: NUM_HEADS,
            num_encoders: NUM_ENCODERS,
            patch_size: PATCH_SIZE,
            hidden_dim: HIDDEN_DIM,
            dropout: DROPOUT,
            sinkhorn_temp: SINKHORN_TEMP as f32,
            activation: "gelu".to_string(),
        }
    }

    #[test]
    fn test_vit() {
        let device = Device::default();
        let test_image = Tensor::<B, 4>::zeros(
            Shape::new([BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE]),
            &device,
        );
        let model = test_config().init::<B>(&device, IN_CHANNELS, IMG_SIZE, NUM_CLASSES);
        let vit_output = model.forward(test_image);
        assert_eq!(vit_output.shape(), Shape::new([BATCH_SIZE, NUM_CLASSES]));
    }
}
