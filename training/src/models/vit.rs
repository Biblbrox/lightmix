use burn::{
    backend::Autodiff,
    module::Module,
    nn::{
        LayerNorm, LayerNormConfig, Linear, LinearConfig,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

use crate::{
    data::batch::ImageBatch,
    models::ModelConfig,
    tokenization::vit::{PatchEmbedding, PatchEmbeddingConfig},
};

/// Standard ViT implementation with cls token and fixed
/// embed_dim
#[derive(Module, Debug)]
pub struct ViT<B: Backend> {
    embedding_block: PatchEmbedding<B>,
    encoder: TransformerEncoder<B>,
    layer_norm: LayerNorm<B>,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ViTConfig {
    in_channels: usize,
    embed_dim: usize,
    num_heads: usize,
    num_layers: usize,
    num_classes: usize,
    patch_size: usize,
    image_size: usize,
    dropout: f64,
}

impl<B: Backend> ViT<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.embedding_block.forward(images);
        let encoder_input = TransformerEncoderInput::new(x);
        let x = self.encoder.forward(encoder_input);
        let x = self.layer_norm.forward(x);
        self.linear.forward(x.slice(s![.., 0, ..])).squeeze() // [batch_size, num_classes]
    }
}

impl ViTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ViT<B> {
        let num_patches = self.patch_size.pow(2);
        ViT {
            embedding_block: PatchEmbeddingConfig::new(
                self.in_channels,
                self.embed_dim,
                self.patch_size,
                self.image_size,
                self.dropout,
                num_patches,
                true,
            )
            .init(device),
            encoder: TransformerEncoderConfig::new(
                self.embed_dim,
                self.embed_dim,
                self.num_heads,
                self.num_layers,
            )
            .with_norm_first(true)
            .with_dropout(self.dropout)
            .init(device),
            layer_norm: LayerNormConfig::new(self.embed_dim).init(device),
            linear: LinearConfig::new(self.embed_dim, self.num_classes).init(device),
        }
    }
}

impl<B: Backend> ModelConfig<B> for ViTConfig {
    type TrainModel = ViT<Autodiff<B>>;
    type ValidModel = ViT<B>;

    fn init_training(&self, device: &B::Device) -> Self::TrainModel {
        self.init(device)
    }

    fn init_inference(&self, device: &B::Device) -> Self::ValidModel {
        self.init(device)
    }
}

impl<B: Backend> ViT<B> {
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

impl<B: AutodiffBackend> TrainStep for ViT<B> {
    type Input = ImageBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: ImageBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for ViT<B> {
    type Input = ImageBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: ImageBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[cfg(test)]
mod tests {
    use burn_cuda::Cuda;

    use crate::tokenization::vit::PatcherConfig;

    use super::*;

    type Backend = Cuda<f32, i32>;

    const PATCH_SIZE: usize = 4;
    const IMG_SIZE: usize = 32;
    const NUM_PATCHES: usize = (IMG_SIZE / PATCH_SIZE).pow(2); // 64
    const EMBED_DIM: usize = PATCH_SIZE.pow(2) * 1;
    const NUM_HEADS: usize = 8;
    const NUM_ENCODERS: usize = 4;
    const NUM_CLASSES: usize = 10;
    const BATCH_SIZE: usize = 10;
    const NUM_CHANNELS: usize = 1;

    #[test]
    fn test_patcher() {
        let device = burn::backend::cuda::CudaDevice::default();

        // Create test image
        let test_image = Tensor::<Backend, 4>::zeros(
            Shape::new([BATCH_SIZE, NUM_CHANNELS, IMG_SIZE, IMG_SIZE]),
            &device,
        );

        // Create pather
        let patcher = PatcherConfig::new(NUM_CHANNELS, EMBED_DIM, PATCH_SIZE).init(&device);
        let patched_image = patcher.forward(test_image);

        assert_eq!(
            patched_image.shape(),
            Shape::new([BATCH_SIZE, NUM_PATCHES, EMBED_DIM])
        );
    }

    #[test]
    fn test_patch_embedding() {
        let device = burn::backend::cuda::CudaDevice::default();

        // Create test image
        let test_image = Tensor::<Backend, 4>::zeros(
            Shape::new([BATCH_SIZE, NUM_CHANNELS, IMG_SIZE, IMG_SIZE]),
            &device,
        );

        let model = PatchEmbeddingConfig::new(
            NUM_CHANNELS,
            EMBED_DIM,
            PATCH_SIZE,
            IMG_SIZE,
            0.1,
            NUM_PATCHES,
            true,
        )
        .init(&device);
        let vit_input = model.forward(test_image);
        assert_eq!(
            vit_input.shape(),
            Shape::new([BATCH_SIZE, NUM_PATCHES + 1, EMBED_DIM])
        );
    }

    #[test]
    fn test_vit() {
        let device = burn::backend::cuda::CudaDevice::default();

        // Create test image
        let test_image = Tensor::<Backend, 4>::zeros(
            Shape::new([BATCH_SIZE, NUM_CHANNELS, IMG_SIZE, IMG_SIZE]),
            &device,
        );

        let model = ViTConfig::new(
            NUM_CHANNELS,
            EMBED_DIM,
            NUM_HEADS,
            NUM_ENCODERS,
            NUM_CLASSES,
            PATCH_SIZE,
            IMG_SIZE,
            0.1,
        )
        .init(&device);
        let vit_output = model.forward(test_image);
        assert_eq!(vit_output.shape(), Shape::new([BATCH_SIZE, NUM_CLASSES]));
    }
}
