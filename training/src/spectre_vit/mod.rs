mod benchmark;
mod embeddings;
mod permute;

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
    },
    prelude::Backend,
    tensor::s,
};

use crate::{
    norm::{DynamicERF, DynamicERFConfig},
    spectre_vit::{
        embeddings::{SpectrePatchEmbedding, SpectrePatchEmbeddingConfig},
        permute::{MHPermutMix, MHPermutMixConfig, MHPermutMixMatrix, MHPermutMixMatrixConfig},
    },
};

#[derive(Module, Debug)]
pub struct SpectreLinear<B: Backend> {
    linear: Linear<B>,
    avg_pool: AdaptiveAvgPool1d,
    norm: DynamicERF<B>,
    activation: Gelu,
}

#[derive(Config, Debug)]
pub struct SpectreLinearConfig {
    in_channels: usize,
    out_channels: usize,
}

#[derive(Module, Debug)]
pub struct SpectreEncoderLayer<B: Backend> {
    linear1: SpectreLinear<B>,
    linear2: SpectreLinear<B>,
    //linear1: Linear<B>,
    //linear2: Linear<B>,
    mix_layer: MHPermutMixMatrix<B>,
    norm1: DynamicERF<B>,
    norm2: DynamicERF<B>,

    dropout1: Dropout,
    dropout2: Dropout,
    activation: Gelu
}

#[derive(Config, Debug)]
pub struct SpectreEncoderLayerConfig {
    seq_length: usize,
    embed_dim: usize,
    num_heads: usize,
    hidden_dim: usize,
    dropout: f64,
    activation: String,
    num_encoders: usize,
    encoder: usize
}

#[derive(Module, Debug)]
pub struct SpectreEncoder<B: Backend> {
    encoder_layers: Vec<SpectreEncoderLayer<B>>,
    norm: Option<DynamicERF<B>>,
}

#[derive(Config, Debug)]
pub struct SpectreEncoderConfig {
    num_layers: usize,
    seq_length: usize,
    embed_dim: usize,
    num_heads: usize,
    hid_dim: usize,
    dropout: f64,
    activation: String,
}

#[derive(Module, Debug)]
pub struct SpectreViT<B: Backend> {
    embedding_block: SpectrePatchEmbedding<B>,
    encoder: SpectreEncoder<B>,
    layer_norm: DynamicERF<B>,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct SpectreViTConfig {
    in_channels: usize,
    embed_dim: usize,
    num_heads: usize,
    num_layers: usize,
    num_classes: usize,
    patch_size: usize,
    image_size: usize,
    hid_dim: usize,
    dropout: f64,
}

impl<B: Backend> SpectreLinear<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let feat = self
            .activation
            .forward(self.norm.forward(self.linear.forward(x.clone())));
        feat + self.avg_pool.forward(x)
        //self.activation
        //    .forward(self.norm.forward(self.linear.forward(x)))
    }
}

impl SpectreLinearConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectreLinear<B> {
        SpectreLinear {
            linear: LinearConfig::new(self.in_channels, self.out_channels).init(device),
            norm: DynamicERFConfig::new(self.out_channels, 0.5, 0.0).init(device),
            activation: Gelu::new(),
            avg_pool: AdaptiveAvgPool1dConfig::new(self.out_channels).init(),
        }
    }
}

impl<B: Backend> SpectreEncoderLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.norm1.forward(self.mix_layer.forward(x.clone()));// + x;
        //let x = self.mix_layer.forward(x.clone()) + x;
        self.norm2.forward(x.clone() + self._ff_block(x))
    }

    pub fn _ff_block(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.dropout1.forward(self.linear1.forward(x));
        self.dropout2.forward(self.linear2.forward(x))
    }
}

impl SpectreEncoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectreEncoderLayer<B> {
        SpectreEncoderLayer {
            linear1: SpectreLinearConfig::new(self.embed_dim, self.hidden_dim).init(device),
            linear2: SpectreLinearConfig::new(self.hidden_dim, self.embed_dim).init(device),
            mix_layer: MHPermutMixMatrixConfig::new(
                self.embed_dim,
                self.seq_length,
                self.num_heads,
                self.embed_dim,
                self.num_encoders,
                self.encoder,
            )
            .init(device),
            norm1: DynamicERFConfig::new(self.embed_dim, 0.5, 0.0).init(device),
            norm2: DynamicERFConfig::new(self.embed_dim, 0.5, 0.0).init(device),
            dropout1: DropoutConfig::new(self.dropout).init(),
            dropout2: DropoutConfig::new(self.dropout).init(),
            activation: Gelu
        }
    }
}

impl<B: Backend> SpectreEncoder<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut output = x.clone();
        for layer in self.encoder_layers.iter() {
            output = layer.forward(output);
        }

        if let Some(norm) = &self.norm {
            output = norm.forward(output);
        }

        output + x
    }
}

impl SpectreEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectreEncoder<B> {
        let mut layers = Vec::new();

        for encoder in 0..self.num_layers {
            layers.push(
                SpectreEncoderLayerConfig::new(
                    self.seq_length,
                    self.embed_dim,
                    self.num_heads,
                    self.hid_dim,
                    self.dropout,
                    self.activation.clone(),
                    self.num_layers,
                    encoder
                )
                .init(device),
            );
        }
        SpectreEncoder {
            encoder_layers: layers,
            norm: Option::None,
        }
    }
}

impl<B: Backend> SpectreViT<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.embedding_block.forward(images);
        let x = self.encoder.forward(x);
        let x = self.layer_norm.forward(x);
        self.linear.forward(x.slice(s![.., 0, ..])).squeeze() // [batch_size, num_classes]
    }
}

impl SpectreViTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectreViT<B> {
        let num_patches = (self.image_size / self.patch_size).pow(2);
        SpectreViT {
            embedding_block: SpectrePatchEmbeddingConfig::new(
                self.in_channels,
                self.embed_dim,
                self.patch_size,
                self.image_size,
                self.dropout
            )
            .init(device),

            encoder: SpectreEncoderConfig::new(
                self.num_layers,
                num_patches + 1,
                self.embed_dim,
                self.num_heads,
                self.hid_dim,
                self.dropout,
                "relu".to_string(),
            )
            .init(device),
            layer_norm: DynamicERFConfig::new(self.embed_dim, 0.5, 0.0).init(device),
            linear: LinearConfig::new(self.embed_dim, self.num_classes).init(device),
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Shape;

    use crate::spectre_vit::embeddings::SpectrePatcherConfig;

    use super::*;

    const IN_CHANNELS: usize = 3;
    const PATCH_SIZE: usize = 4;
    const IMG_SIZE: usize = 32;
    const NUM_PATCHES: usize = (IMG_SIZE / PATCH_SIZE).pow(2); // 64
    const EMBED_DIM: usize = PATCH_SIZE.pow(2) * 1;
    const NUM_HEADS: usize = 8;
    const NUM_ENCODERS: usize = 4;
    const NUM_CLASSES: usize = 10;
    const BATCH_SIZE: usize = 10;
    const NUM_CHANNELS: usize = 1;
    const HIDDEN_DIM: usize = 64;
    const DROPOUT: f64 = 0.1;

    #[test]
    fn test_patcher() {
        type B = burn::backend::cuda::Cuda;
        let device = burn::backend::cuda::CudaDevice::default();

        // Create test image
        let test_image = Tensor::<B, 4>::zeros(
            Shape::new([BATCH_SIZE, NUM_CHANNELS, IMG_SIZE, IMG_SIZE]),
            &device,
        );

        // Create pather
        let patcher = SpectrePatcherConfig::new(IN_CHANNELS, EMBED_DIM, PATCH_SIZE).init(&device);
        let patched_image = patcher.forward(test_image);

        assert_eq!(
            patched_image.shape(),
            Shape::new([BATCH_SIZE, NUM_PATCHES, EMBED_DIM])
        );
    }

    #[test]
    fn test_patch_embedding() {
        type B = burn::backend::cuda::Cuda;
        let device = burn::backend::cuda::CudaDevice::default();

        // Create test image
        let test_image = Tensor::<B, 4>::zeros(
            Shape::new([BATCH_SIZE, NUM_CHANNELS, IMG_SIZE, IMG_SIZE]),
            &device,
        );

        let model = SpectrePatchEmbeddingConfig::new(IN_CHANNELS, EMBED_DIM, PATCH_SIZE, IMG_SIZE, DROPOUT)
            .init(&device);
        let vit_input = model.forward(test_image);
        assert_eq!(
            vit_input.shape(),
            Shape::new([BATCH_SIZE, NUM_PATCHES + 1, EMBED_DIM])
        );
    }

    #[test]
    fn test_vit() {
        type B = burn::backend::cuda::Cuda;
        let device = burn::backend::cuda::CudaDevice::default();

        // Create test image
        let test_image = Tensor::<B, 4>::zeros(
            Shape::new([BATCH_SIZE, NUM_CHANNELS, IMG_SIZE, IMG_SIZE]),
            &device,
        );

        let model = SpectreViTConfig::new(
            IN_CHANNELS,
            EMBED_DIM,
            NUM_HEADS,
            NUM_ENCODERS,
            NUM_CLASSES,
            PATCH_SIZE,
            IMG_SIZE,
            HIDDEN_DIM,
            DROPOUT,
        )
        .init(&device);
        let vit_output = model.forward(test_image);
        assert_eq!(vit_output.shape(), Shape::new([BATCH_SIZE, NUM_CLASSES]));
    }
}
