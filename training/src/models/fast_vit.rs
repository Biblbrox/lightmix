use burn::{
    Tensor,
    backend::Autodiff,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, loss::CrossEntropyLossConfig},
    tensor::{
        Int,
        backend::{AutodiffBackend, Backend},
    },
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

use crate::{
    data::batch::ImageBatch,
    encoders::fast_encoder::{FastEncoder, FastEncoderConfig},
    models::ModelConfig,
    norm::{DynamicERF, DynamicERFConfig},
    tokenization::vit::{PatchEmbedding, PatchEmbeddingConfig},
};

#[derive(Module, Debug)]
pub struct FastViT<B: Backend> {
    embedding_block: PatchEmbedding<B>,
    encoder: FastEncoder<B>,
    layer_norm: DynamicERF<B>,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct FastViTConfig {
    in_channels: usize,
    embed_dim: usize,
    num_heads: usize,
    num_layers: usize,
    num_classes: usize,
    patch_size: usize,
    image_size: usize,
    hid_dim: usize,
    dropout: f64,
    sinkhorn_temp: f32,
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> FastViT<B> {
        let grid_size = self.image_size / self.patch_size;
        let num_patches = grid_size.pow(2);

        FastViT {
            embedding_block: PatchEmbeddingConfig::new(
                self.in_channels,
                self.embed_dim,
                self.patch_size,
                self.image_size,
                self.dropout,
                num_patches,
                false,
            )
            .init(device),

            encoder: FastEncoderConfig::new(
                self.num_layers,
                grid_size,
                num_patches,
                self.embed_dim,
                self.num_heads,
                self.hid_dim,
                self.dropout,
                self.sinkhorn_temp,
            )
            .init(device),
            layer_norm: DynamicERFConfig::new(self.embed_dim).init(device),
            linear: LinearConfig::new(self.embed_dim, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> ModelConfig<B> for FastViTConfig {
    type TrainModel = FastViT<Autodiff<B>>;
    type ValidModel = FastViT<B>;

    fn init_training(&self, device: &B::Device) -> Self::TrainModel {
        self.init(device)
    }

    fn init_inference(&self, device: &B::Device) -> Self::ValidModel {
        self.init(device)
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
    type Input = ImageBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: ImageBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for FastViT<B> {
    type Input = ImageBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: ImageBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
