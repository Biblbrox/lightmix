use burn::{
    module::Module,
    nn::{
        LayerNorm, LayerNormConfig, Linear, LinearConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
};

use crate::tokenization::vit::{PatchEmbedding, PatchEmbeddingConfig};

#[derive(Module, Debug)]
pub struct ViT<B: Backend> {
    embedding_block: PatchEmbedding<B>,
    encoder: TransformerEncoder<B>,
    layer_norm: LayerNorm<B>,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ViTConfig {
    embed_dim: usize,
    num_heads: usize,
    num_layers: usize,
    num_classes: usize,
    patch_size: usize,
    image_size: usize,
}

impl<B: Backend> ViT<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.embedding_block.forward(images);
        let batch_size = x.dims()[0];
        let seq_length = x.dims()[1];
        //let mask_attn = generate_autoregressive_mask(batch_size, seq_length, &x.device());
        let encoder_input = TransformerEncoderInput::new(x);
        let x = self.encoder.forward(encoder_input);
        let x = self.layer_norm.forward(x);
        self.linear.forward(x.slice(s![.., 0, ..])).squeeze() // [batch_size, num_classes]
    }
}

impl ViTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ViT<B> {
        ViT {
            embedding_block: PatchEmbeddingConfig::new(
                self.embed_dim,
                self.patch_size,
                self.image_size,
            )
            .init(device),
            encoder: TransformerEncoderConfig::new(
                self.embed_dim,
                self.embed_dim,
                self.num_heads,
                self.num_layers,
            )
            .with_norm_first(true)
            .with_dropout(0.001)
            .init(device),
            layer_norm: LayerNormConfig::new(self.embed_dim).init(device),
            linear: LinearConfig::new(self.embed_dim, self.num_classes).init(device),
        }
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
        let patcher = PatcherConfig::new(EMBED_DIM, PATCH_SIZE).init(&device);
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

        let model = PatchEmbeddingConfig::new(EMBED_DIM, PATCH_SIZE, IMG_SIZE).init(&device);
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
            EMBED_DIM,
            NUM_HEADS,
            NUM_ENCODERS,
            NUM_CLASSES,
            PATCH_SIZE,
            IMG_SIZE,
        )
        .init(&device);
        let vit_output = model.forward(test_image);
        assert_eq!(vit_output.shape(), Shape::new([BATCH_SIZE, NUM_CLASSES]));
    }
}
