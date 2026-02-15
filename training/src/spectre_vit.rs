use std::ops::Range;

use burn::{
    module::{Module, Param, Parameter},
    nn::{
        Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        attention::generate_autoregressive_mask,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
        transformer::{
            TransformerDecoderLayer, TransformerEncoder, TransformerEncoderConfig,
            TransformerEncoderInput, TransformerEncoderLayer,
        },
    },
    prelude::*,
    tensor::{Distribution, linalg::Norm},
};

// Patch Embedding Layer
#[derive(Module, Debug)]
pub struct Patcher<B: Backend> {
    conv: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct PatcherConfig {
    embed_dim: usize,
    patch_size: usize,
}

#[derive(Module, Debug)]
pub struct PatchEmbedding<B: Backend> {
    patcher: Patcher<B>,
    cls_token: Param<Tensor<B, 3>>,
    position_embeddings: Param<Tensor<B, 3>>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct PatchEmbeddingConfig {
    embed_dim: usize,
    patch_size: usize,
    image_size: usize,
}

#[derive(Module, Debug)]
pub struct MHPermutMix<B: Backend> {
    signs: Vec<Vec<Tensor<B, 1>>>,
    perms: Vec<Vec<Tensor<B, 1, Int>>>,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct MHPermutMixConfig {
    embed_dim: usize,
    seq_length: usize,
    num_heads: usize,
    out_channels: usize,
    num_encoders: usize,
}

#[derive(Module, Debug)]
pub struct SpectreLinear<B: Backend> {
    linear: Linear<B>,
    avg_pool: AdaptiveAvgPool1d,
    norm: LayerNorm<B>,
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
    mix_layer: MHPermutMix<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,

    dropout1: Dropout,
    dropout2: Dropout,
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
}

#[derive(Module, Debug)]
pub struct SpectreEncoder<B: Backend> {
    encoder_layers: Vec<SpectreEncoderLayer<B>>,
    norm: Option<LayerNorm<B>>,
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
    embedding_block: PatchEmbedding<B>,
    encoder: SpectreEncoder<B>,
    layer_norm: LayerNorm<B>,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct SpectreViTConfig {
    embed_dim: usize,
    num_heads: usize,
    num_layers: usize,
    num_classes: usize,
    patch_size: usize,
    image_size: usize,
    hid_dim: usize,
    dropout: f64,
}

impl<B: Backend> Patcher<B> {
    // # Shapes
    // - Images: [batch_size, num_channels, height, width]
    // - Output: [batch_size, num_channels, num_patches + 1, embed_dim]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 3> {
        let x = self.conv.forward(images); // [batch_size, embed_dim, row_patch_num, row_patch_num]
        let x = x.flatten(2, 3); // [batch_suze, embed_dim, total_patch_num]
        x.swap_dims(1, 2) // [batch_size, total_patch_num, embed_dim]
    }
}

impl PatcherConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Patcher<B> {
        Patcher {
            conv: Conv2dConfig::new([1, self.embed_dim], [self.patch_size, self.patch_size])
                .with_stride([self.patch_size, self.patch_size])
                .init(device),
        }
    }
}

impl<B: Backend> PatchEmbedding<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 3> {
        // TODO: Bug is here
        let patches = self.patcher.forward(images.clone()); // [batch_size, total_patch_dim, embed_dim]
        // Expand cls_token alongside batch dimension. Left other
        // dimensions untouched
        let cls_token_batch = self
            .cls_token
            .val()
            .expand([images.dims()[0] as i32, -1, -1]);
        // Concatenate cls token and image patches
        let x = Tensor::cat(Vec::from([cls_token_batch, patches]), 1);
        let x = self.position_embeddings.val() + x;
        let x = self.dropout.forward(x);
        x // [batch_size, total_patch_dim + 1, embed_dim]
    }
}

impl PatchEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PatchEmbedding<B> {
        let distribution = Distribution::Normal(0.0, 1.0);
        let num_patches = (self.image_size / self.patch_size).pow(2);
        PatchEmbedding {
            patcher: PatcherConfig::new(self.embed_dim, self.patch_size).init(device),
            cls_token: Param::<Tensor<B, 3>>::from_tensor(Tensor::<B, 3>::random(
                Shape::new([1, 1, self.embed_dim]),
                distribution,
                device,
            ))
            .set_require_grad(true),
            position_embeddings: Param::<Tensor<B, 3>>::from_tensor(Tensor::<B, 3>::random(
                Shape::new([1, num_patches + 1, self.embed_dim]),
                distribution,
                device,
            ))
            .set_require_grad(true),
            dropout: DropoutConfig::new(0.001).init(),
        }
    }
}

impl<B: Backend> SpectreLinear<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let feat = self
            .activation
            .forward(self.norm.forward(self.linear.forward(x.clone())));
        return feat + self.avg_pool.forward(x);
    }
}

impl SpectreLinearConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectreLinear<B> {
        SpectreLinear {
            linear: LinearConfig::new(self.in_channels, self.out_channels).init(device),
            norm: LayerNormConfig::new(self.out_channels).init(device),
            activation: Gelu::new(),
            avg_pool: AdaptiveAvgPool1dConfig::new(self.out_channels).init(),
        }
    }
}

impl<B: Backend> MHPermutMix<B> {
    pub fn forward(&self, x: Tensor<B, 3>, encoder_num: usize) -> Tensor<B, 3> {
        let shape = x.shape(); // [B, N, E]
        let x = x.reshape([shape[0], shape[1] * shape[2]]); // [B, N * E]
        assert_eq!(x.shape(), Shape::new([shape[0], shape[1] * shape[2]]));
        let mut fused_permuted = Vec::<Tensor<B, 3>>::new();
        for i in 0..self.signs[0].len() {
            let transform = self.perms[encoder_num][i].clone(); // [N * E]
            assert_eq!(transform.shape(), Shape::new([shape[1] * shape[2]]));
            let signs = self.signs[encoder_num][i].clone();
            assert_eq!(signs.shape(), Shape::new([shape[1] * shape[2]]));
            let permuted = x.clone().select(1, transform) * signs.unsqueeze();
            let permuted = permuted.reshape(shape.clone());
            fused_permuted.push(permuted);
        }
        let x = Tensor::<B, 3>::cat(fused_permuted, 2);
        assert_eq!(
            x.shape(),
            Shape::new([shape[0], shape[1], shape[2] * self.signs[0].len()])
        );
        let x = self.linear.forward(x);
        return x;
    }
}

impl MHPermutMixConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MHPermutMix<B> {
        let distribution = Distribution::Uniform(-1.0, 1.0);
        let d = self.embed_dim * self.seq_length;
        let mut perms_per_encoder = Vec::<Vec<Tensor<B, 1, Int>>>::new();
        let mut sign_per_encoder = Vec::<Vec<Tensor<B, 1>>>::new();
        (0..self.num_encoders).for_each(|_| {
            let mut perms_per_head = Vec::<Tensor<B, 1, Int>>::new();
            let mut sign_per_head = Vec::<Tensor<B, 1>>::new();
            (0..self.num_heads).for_each(|_| {
                let rand = Tensor::<B, 1>::random(
                    Shape::new([d]),
                    Distribution::Uniform(0.0, 1.0),
                    device,
                );

                perms_per_head.push(rand.argsort(0).set_require_grad(false));
                sign_per_head.push(
                    Tensor::<B, 1>::random(
                        Shape::new([self.embed_dim * self.seq_length]),
                        distribution,
                        device,
                    )
                    .sign()
                    .set_require_grad(false),
                )
            });
            perms_per_encoder.push(perms_per_head);
            sign_per_encoder.push(sign_per_head);
        });

        MHPermutMix {
            signs: sign_per_encoder,
            perms: perms_per_encoder,
            linear: LinearConfig::new(self.embed_dim * self.num_heads, self.out_channels)
                .init(device),
        }
    }
}

impl<B: Backend> SpectreEncoderLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>, encoder_num: usize) -> Tensor<B, 3> {
        let x = self
            .norm1
            .forward(self.mix_layer.forward(x.clone(), encoder_num))
            + x;
        let x = self.norm2.forward(x.clone() + self._ff_block(x.clone()));
        return x;
    }

    pub fn _ff_block(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.dropout1.forward(self.linear1.forward(x));
        let x = self.dropout2.forward(self.linear2.forward(x));
        return x;
    }
}

impl SpectreEncoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectreEncoderLayer<B> {
        SpectreEncoderLayer {
            linear1: SpectreLinearConfig::new(self.embed_dim, self.hidden_dim).init(device),
            linear2: SpectreLinearConfig::new(self.hidden_dim, self.embed_dim).init(device),
            mix_layer: MHPermutMixConfig::new(
                self.embed_dim,
                self.seq_length,
                self.num_heads,
                self.embed_dim,
                self.num_encoders,
            )
            .init(device),
            norm1: LayerNormConfig::new(self.embed_dim).init(device),
            norm2: LayerNormConfig::new(self.embed_dim).init(device),
            dropout1: DropoutConfig::new(self.dropout).init(),
            dropout2: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> SpectreEncoder<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut output = x.clone();
        for (idx, layer) in self.encoder_layers.iter().enumerate() {
            output = layer.forward(output, idx);
        }

        if !self.norm.as_ref().is_none() {
            output = self.norm.as_ref().unwrap().forward(output);
        }

        return output + x.clone();
    }
}

impl SpectreEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectreEncoder<B> {
        let mut layers = Vec::new();

        for _ in 0..self.num_layers {
            layers.push(
                SpectreEncoderLayerConfig::new(
                    self.seq_length,
                    self.embed_dim,
                    self.num_heads,
                    self.hid_dim,
                    self.dropout,
                    self.activation.clone(),
                    self.num_layers,
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
            embedding_block: PatchEmbeddingConfig::new(
                self.embed_dim,
                self.patch_size,
                self.image_size,
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
            layer_norm: LayerNormConfig::new(self.embed_dim).init(device),
            linear: LinearConfig::new(self.embed_dim, self.num_classes).init(device),
        }
    }
}

#[cfg(test)]
mod tests {
    use burn_cuda::Cuda;

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
    const HIDDEN_DIM: usize = 64;
    const DROPOUT: f64 = 0.1;

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

        let model = SpectreViTConfig::new(
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
