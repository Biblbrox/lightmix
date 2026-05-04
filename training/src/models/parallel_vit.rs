use burn::{
    Tensor,
    backend::Autodiff,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, loss::CrossEntropyLossConfig},
    prelude::Backend,
    tensor::{Int, backend::AutodiffBackend},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

use crate::{
    data::batch::ImageBatch,
    encoders::fast_encoder::{FastEncoderLayer, FastEncoderLayerConfig},
    models::ModelConfig,
    norm::{DynamicERF, DynamicERFConfig},
    tokenization::vit::{PatchEmbedding, PatchEmbeddingConfig},
};

/// Trial to reduce computantional cost of ViT by making execution of
/// some layers parallel
#[derive(Module, Debug)]
pub struct ParallelEncoder<B: Backend> {
    /// seq_Layers executed sequentially.
    seq_layers: Vec<FastEncoderLayer<B>>,
    /// seq_Layers executed in parallel with further concatenation
    par_layers: Vec<FastEncoderLayer<B>>,
    norm: Option<DynamicERF<B>>,
}

#[derive(Config, Debug)]
pub struct ParallelEncoderConfig {
    num_layers: usize,
    grid_size: usize,
    seq_length: usize,
    embed_dim: usize,
    num_heads: usize,
    hid_dim: usize,
    dropout: f64,
    activation: String,
    sinkhorn_temp: f32,
}

#[derive(Module, Debug)]
pub struct ParallelViT<B: Backend> {
    embedding_block: PatchEmbedding<B>,
    encoder: ParallelEncoder<B>,
    layer_norm: DynamicERF<B>,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ParallelViTConfig {
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

impl<B: Backend> ParallelEncoder<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Parallel execution with concatenation of the outputs in embedding space
        let mut output_par = vec![];
        for layer in self.par_layers.iter() {
            output_par.push(layer.forward(x.clone()));
        }

        let mut output = Tensor::cat(output_par, 2);

        // Sequential execution
        for layer in self.seq_layers.iter() {
            output = layer.forward(output);
        }

        if let Some(norm) = &self.norm {
            output = norm.forward(output);
        }

        output
    }
}

impl ParallelEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ParallelEncoder<B> {
        let mut seq_layers = Vec::new();
        let mut par_layers = Vec::new();

        let seq_embed_dim = self.num_layers / 2 * self.embed_dim;
        let par_embed_dim = self.embed_dim;
        for i in 0..self.num_layers / 2 {
            seq_layers.push(
                FastEncoderLayerConfig::new(
                    self.seq_length,
                    seq_embed_dim,
                    self.num_heads,
                    self.hid_dim,
                    self.dropout,
                    self.activation.clone(),
                    self.num_layers,
                    i,
                    self.sinkhorn_temp,
                )
                .init(device),
            );
            par_layers.push(
                FastEncoderLayerConfig::new(
                    self.seq_length,
                    par_embed_dim,
                    self.num_heads,
                    self.hid_dim,
                    self.dropout,
                    self.activation.clone(),
                    self.num_layers,
                    self.num_layers / 2 + i,
                    self.sinkhorn_temp,
                )
                .init(device),
            );
        }
        ParallelEncoder {
            seq_layers,
            par_layers,
            norm: Option::None,
        }
    }
}

impl<B: Backend> ParallelViT<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.embedding_block.forward(images);
        let x = self.encoder.forward(x);
        let x = self.layer_norm.forward(x);

        self.linear.forward(x.mean_dim(1)).squeeze()
    }
}

impl ParallelViTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ParallelViT<B> {
        let grid_size = self.image_size / self.patch_size;
        let num_patches = grid_size.pow(2);

        ParallelViT {
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

            encoder: ParallelEncoderConfig::new(
                self.num_layers,
                grid_size,
                num_patches,
                self.embed_dim,
                self.num_heads,
                self.hid_dim,
                self.dropout,
                "relu".to_string(),
                self.sinkhorn_temp,
            )
            .init(device),
            layer_norm: DynamicERFConfig::new(self.embed_dim * self.num_layers / 2).init(device),
            linear: LinearConfig::new(self.embed_dim * self.num_layers / 2, self.num_classes)
                .init(device),
        }
    }
}

impl<B: Backend> ModelConfig<B> for ParallelViTConfig {
    type TrainModel = ParallelViT<Autodiff<B>>;
    type ValidModel = ParallelViT<B>;

    fn init_training(&self, device: &B::Device) -> Self::TrainModel {
        self.init(device)
    }

    fn init_inference(&self, device: &B::Device) -> Self::ValidModel {
        self.init(device)
    }
}

impl<B: Backend> ParallelViT<B> {
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

impl<B: AutodiffBackend> TrainStep for ParallelViT<B> {
    type Input = ImageBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: ImageBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for ParallelViT<B> {
    type Input = ImageBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: ImageBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
