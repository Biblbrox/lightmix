use std::ops::Range;

use burn::{
    module::{Module, Param},
    nn::{
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        attention::generate_autoregressive_mask,
        conv::{Conv2d, Conv2dConfig},
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
    signs: Tensor<B, 3>,
    perms: Tensor<B, 3, Int>,
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
pub struct SpectreEncoderLayer<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
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
    norm: Option<LayerNorm<B>>
}

#[derive(Config, Debug)]
pub struct SpectreEncoderConfig<B: Backend> {
    num_layers: usize,
    encoder_layer: SpectreEncoderLayer<B>
}

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
}erive(Debug)] to SpectreEncoder<B> or manually impl Debug for SpectreEncoder<B> (rustc E0277)

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

impl<B: Backend> MHPermutMix<B> {
    pub fn forward(&self, x: Tensor<B, 3>, encoder_num: usize) -> Tensor<B, 3> {
        let shape = x.shape(); // [B, N, E]
        let x = x.reshape([shape[0], shape[1] * shape[2]]); // [B, N * E]
        let head_transforms: Tensor<B, 2, Int> = self
            .perms
            .clone()
            .slice([encoder_num..encoder_num])
            .squeeze(); // [H, N * E]
        let head_signs: Tensor<B, 2> = self
            .signs
            .clone()
            .slice([encoder_num..encoder_num])
            .squeeze();
        let mut fused_permuted = Vec::<Tensor<B, 3>>::new();
        for i in 0..head_transforms.shape()[1] {
            let transform = head_transforms.clone().slice(i..i).squeeze(); // [N * E]
            let signs = head_signs.clone().slice(i..i).squeeze();
            let permuted = x.clone().select(1, transform) * signs;
            let permuted = permuted.reshape(shape.clone());
            fused_permuted.push(permuted);
        }
        let fused = Tensor::<B, 3>::cat(fused_permuted, 0);
        return fused;
    }
}

impl MHPermutMixConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MHPermutMix<B> {
        let distribution = Distribution::Uniform(-1.0, 1.0);
        let d = self.embed_dim * self.seq_length;
        let mut perms_per_encoder = Vec::<Tensor<B, 2, Int>>::new();
        (0..self.num_encoders).for_each(|_| {
            let mut perms_per_head = Vec::<Tensor<B, 1, Int>>::new();
            (0..self.num_heads).for_each(|_| {
                perms_per_head.push(Tensor::<B, 1, Int>::arange(0..d.to_i64(), device));
            });
            perms_per_encoder.push(Tensor::<B, 1, Int>::stack(perms_per_head, 0));
        });
        let perms_per_encoder = Tensor::<B, 2, Int>::stack(perms_per_encoder, 0);

        MHPermutMix {
            signs: Tensor::<B, 3>::random(
                Shape::new([
                    self.num_encoders,
                    self.num_heads,
                    self.embed_dim * self.seq_length,
                ]),
                distribution,
                device,
            )
            .sign()
            .set_require_grad(false),
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
            linear1: LinearConfig::new(self.embed_dim, self.hidden_dim).init(device),
            linear2: LinearConfig::new(self.hidden_dim, self.embed_dim).init(device),
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
    pub fn forward(&self, x: Tensor::<B, 3>) -> Tensor<B, 3> {
        let output = x;
        for (idx, layer) in self.encoder_layers.iter().enumerate() {
            output = layer.forward(output, idx);
        }

        if !self.norm.is_none() {
            output = self.norm.unwrap().forward(output);
        }

        return output + x
    }
}

impl SpectreEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Tensor<B, 3> {
        SpectreEncoder {
            encoder_layers: Vec::clone_from(self.encoder_layer),
        }
    }
}

impl<B: Backend> ViT<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.embedding_block.forward(images);
        //let batch_size = x.dims()[0];
        //let seq_length = x.dims()[1];
        //let mask_attn = generate_autoregressive_mask(batch_size, seq_length, &x.device());
        let encoder_input = TransformerEncoderInput::new(x);
        let x = self.encoder.forward(encoder_input);
        let x = self.layer_norm.forward(x);
        self.linear.forward(x.slice(s![.., 0, ..])).squeeze::<2>() // [batch_size, num_classes]
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
