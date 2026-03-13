use burn::{Tensor, prelude::Backend, tensor::Distribution};
use cubecl::{benchmark::Benchmark, future};

use crate::spectre_vit::{
    MHPermutMix, MHPermutMixConfig, SpectreViT, SpectreViTConfig,
    embeddings::{SpectrePatchEmbedding, SpectrePatchEmbeddingConfig},
};

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

pub struct SpectrePatchEmbeddingBenchmark<B: Backend> {
    pub batch_size: usize,
    pub in_channels: usize,
    pub embed_dim: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for SpectrePatchEmbeddingBenchmark<B> {
    type Input = (Tensor<B, 4>, SpectrePatchEmbedding<B>);
    type Output = Tensor<B, 3>;

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
            SpectrePatchEmbeddingConfig::new(
                self.in_channels,
                self.embed_dim,
                self.patch_size,
                self.image_size,
            )
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "SpectrePatchEmbeddingBenchmark-{:?}x{:?}x{:?}x{:?} image",
            self.batch_size, self.in_channels, self.image_size, self.image_size
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, patcher) = input;
        let res = patcher.forward(tensor);
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

#[cfg(test)]
mod tests {
    use burn::backend::Autodiff;
    use cubecl::{
        benchmark::{Benchmark, BenchmarkComputations},
        profile::TimingMethod,
    };

    use crate::{
        spectre_vit::benchmark::{
            MHPermutMixBenchmark, SpectrePatchEmbeddingBenchmark, SpectreViTBenchmark,
        },
        utils::print_bench_results,
    };

    #[test]
    fn spectre_patcher_bench() {
        type B = burn::backend::cuda::Cuda;
        let device = burn::backend::cuda::CudaDevice::default();

        let batches = [8; 5];
        let embed_dim = [64, 128, 256, 512, 1024];
        let image_size: usize = 224;
        let patch_size: usize = 16;
        let in_channels = 3;

        let mut results: Vec<(u32, BenchmarkComputations)> = Vec::new();
        for embed in embed_dim.into_iter() {
            let bench = SpectrePatchEmbeddingBenchmark::<B> {
                batch_size: batches[0],
                in_channels,
                embed_dim: embed,
                patch_size,
                image_size,
                device: device.clone(),
            };

            let bench_res = bench.run(TimingMethod::System).unwrap();
            let computed = BenchmarkComputations::new(&bench_res);
            results.push((embed as u32, computed));
        }

        print_bench_results(&results, "embed_dim");
    }

    #[test]
    fn mh_permute_mix_bench() {
        type B = burn::backend::cuda::Cuda;
        type MyAutodiffBackend = Autodiff<B>;
        let device = burn::backend::cuda::CudaDevice::default();
        let batches = [8; 5];
        let embed_dim = [64, 128, 256, 512, 1024];
        let num_tokens = [65; 5];
        let num_heads = [1, 2, 3, 4, 5, 6, 7, 8];
        let out_channels = [64, 128, 256, 512, 1024];

        let mut results: Vec<(u32, BenchmarkComputations)> = Vec::new();
        for head in num_heads.into_iter() {
            let bench = MHPermutMixBenchmark::<MyAutodiffBackend> {
                batch_size: batches[0],
                embed_dim: embed_dim[0],
                num_heads: head,
                num_tokens: num_tokens[0],
                out_channels: out_channels[0],
                device: device.clone(),
            };

            let bench_res = bench.run(TimingMethod::Device).unwrap();
            let computed = BenchmarkComputations::new(&bench_res);
            results.push((head as u32, computed));
        }

        print_bench_results(&results, "num_heads");
    }

    #[test]
    fn spectre_vit_bench() {
        type B = burn::backend::cuda::Cuda;
        let device = burn::backend::cuda::CudaDevice::default();

        //type B = burn::backend::wgpu::Wgpu;
        //let device = burn::backend::wgpu::WgpuDevice::default();
        let batches = [8; 5];
        let embed_dim = [64, 128, 256, 512, 1024];
        let num_tokens = [65; 5];
        let num_heads = [1, 2, 3, 4, 5, 6, 7, 8];
        let out_channels = [64, 128, 256, 512, 1024];
        let image_size: usize = 224;
        let patch_size: usize = 16;
        let num_patches: usize = (image_size / patch_size).pow(2);
        let num_classes = 1000;
        let hid_dim = 768;
        let in_channels = 3;
        let dropout = 0.5;

        let mut results: Vec<(u32, BenchmarkComputations)> = Vec::new();
        for head in num_heads.into_iter() {
            let bench = SpectreViTBenchmark::<B> {
                num_patches,
                batch_size: batches[0],
                embed_dim: embed_dim[0],
                num_heads: head,
                num_layers: num_tokens[0],
                in_channels,
                num_classes,
                patch_size,
                image_size,
                hid_dim,
                device: device.clone(),
                dropout,
            };

            let bench_res = bench.run(TimingMethod::System).unwrap();
            let computed = BenchmarkComputations::new(&bench_res);
            results.push((head as u32, computed));
        }

        print_bench_results(&results, "num_heads");
    }
}
