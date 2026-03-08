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

use crate::spectre_vit::{
    embeddings::{SpectrePatchEmbedding, SpectrePatchEmbeddingConfig},
    permute::{MHPermutMix, MHPermutMixConfig},
};

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
    embedding_block: SpectrePatchEmbedding<B>,
    encoder: SpectreEncoder<B>,
    layer_norm: LayerNorm<B>,
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
            embedding_block: SpectrePatchEmbeddingConfig::new(
                self.in_channels,
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
    use burn::tensor::Shape;
    use burn_cuda::Cuda;
    use cubecl::{
        benchmark::{Benchmark, BenchmarkComputations},
        cuda::CudaRuntime,
        profile::TimingMethod,
    };

    use crate::spectre_vit::{benchmark::MHPermutMixBenchmark, embeddings::SpectrePatcherConfig};

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

    fn print_markdown_table(results: &[(u8, BenchmarkComputations)]) {
        println!(
            "| {:>10} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} |",
            "num_heads", "mean (µs)", "median (µs)", "variance", "min (µs)", "max (µs)"
        );
        println!(
            "|{:-^12}|{:-^14}|{:-^14}|{:-^14}|{:-^14}|{:-^14}|",
            "", "", "", "", "", ""
        );
        for (heads, c) in results {
            println!(
                "| {:>10} | {:>12.2} | {:>12.2} | {:>12.2} | {:>12.2} | {:>12.2} |",
                heads,
                c.mean.as_micros(),
                c.median.as_micros(),
                c.variance.as_micros(),
                c.min.as_micros(),
                c.max.as_micros(),
            );
        }
    }

    #[test]
    fn mh_permute_mix_bench() {
        type B = burn::backend::cuda::Cuda;
        let device = burn::backend::cuda::CudaDevice::default();
        let batches = vec![8; 5];
        let embed_dim = vec![64, 128, 256, 512, 1024];
        let num_tokens = vec![65; 5];
        let num_heads = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let out_channels = vec![64, 128, 256, 512, 1024];

        let mut results: Vec<(u8, BenchmarkComputations)> = Vec::new();
        for head in num_heads.into_iter() {
            let bench = MHPermutMixBenchmark::<B> {
                batch_size: batches[0],
                embed_dim: embed_dim[0],
                num_heads: head,
                num_tokens: num_tokens[0],
                out_channels: out_channels[0],
                device: device.clone(),
            };

            let bench_res = bench.run(TimingMethod::System).unwrap();
            let computed = BenchmarkComputations::new(&bench_res);
            results.push((head as u8, computed));
        }

        print_markdown_table(&results);
    }

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

        let model = SpectrePatchEmbeddingConfig::new(IN_CHANNELS, EMBED_DIM, PATCH_SIZE, IMG_SIZE)
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
