use burn::{Tensor, prelude::Backend, tensor::Distribution};
use cubecl::{benchmark::Benchmark, future};

use crate::spectre_vit::{MHPermutMix, MHPermutMixConfig, SpectreViT, SpectreViTConfig};

pub struct SpectreViTBenchmark<B: Backend> {
    pub num_patches: usize,
    pub batch_size: usize,
    pub in_channels: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub num_classes: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub hid_dim: usize,
    pub dropout: f64,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for SpectreViTBenchmark<B> {
    type Input = (Tensor<B, 4>, SpectreViT<B>);
    type Output = Tensor<B, 2>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 4>::random(
                [
                    self.batch_size,
                    self.in_channels,
                    self.image_size,
                    self.image_size,
                ],
                Distribution::Default,
                &self.device,
            ),
            SpectreViTConfig::new(
                self.in_channels,
                self.embed_dim,
                self.num_heads,
                self.num_layers,
                self.num_classes,
                self.patch_size,
                self.image_size,
                self.hid_dim,
                self.dropout,
            )
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "SpectreViT-{:?}x{:?}x{:?} {:?} heads",
            self.batch_size, self.num_patches, self.embed_dim, self.num_heads
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, model) = input;
        let res = model.forward(tensor);
        Ok(res)
    }
}

pub struct MHPermutMixBenchmark<B: Backend> {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub out_channels: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for MHPermutMixBenchmark<B> {
    type Input = (Tensor<B, 3>, MHPermutMix<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            MHPermutMixConfig::new(
                self.embed_dim,
                self.num_tokens,
                self.num_heads,
                self.out_channels,
                1,
            )
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "MHPermutMix-{:?}x{:?}x{:?} {:?} heads",
            self.batch_size, self.num_tokens, self.embed_dim, self.num_heads
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, mh_permut) = input;
        let res = mh_permut.forward(tensor, 0);
        Ok(res)
    }
}
