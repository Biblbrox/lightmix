use burn::{Tensor, nn::{Linear, LinearConfig}, prelude::Backend, tensor::Distribution};
use cubecl::{benchmark::Benchmark, future};

use crate::spectre_vit::{
    MHPermutMix, MHPermutMixConfig, SpectreLinear, SpectreLinearConfig, SpectreViT, SpectreViTConfig, embeddings::{SpectrePatchEmbedding, SpectrePatchEmbeddingConfig}, permute::{MHPermutMixMatrix, MHPermutMixMatrixConfig}
};


pub struct SpectreLinearBenchmark<B: Backend> {
    pub batch_size: usize,
    pub num_tokens: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for SpectreLinearBenchmark<B> {
    type Input = (Tensor<B, 3>, SpectreLinear<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [
                    self.batch_size,
                    self.num_tokens,
                    self.in_channels,
                ],
                Distribution::Default,
                &self.device,
            ),
            SpectreLinearConfig::new(
                self.in_channels,
                self.out_channels
            )
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "SpectreLinear-{:?}x{:?}x{:?}",
            self.batch_size, self.num_tokens, self.out_channels
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


pub struct LinearBenchmark<B: Backend> {
    pub batch_size: usize,
    pub num_tokens: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for LinearBenchmark<B> {
    type Input = (Tensor<B, 3>, Linear<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [
                    self.batch_size,
                    self.num_tokens,
                    self.in_channels,
                ],
                Distribution::Default,
                &self.device,
            ),
            LinearConfig::new(
                self.in_channels,
                self.out_channels
            )
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "Linear-{:?}x{:?}x{:?}",
            self.batch_size, self.num_tokens, self.out_channels
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
                0.01,
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
        let res = mh_permut.forward(tensor);
        Ok(res)
    }
}

pub struct MHPermutMixMatrixBenchmark<B: Backend> {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub out_channels: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for MHPermutMixMatrixBenchmark<B> {
    type Input = (Tensor<B, 3>, MHPermutMixMatrix<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            MHPermutMixMatrixConfig::new(
                self.embed_dim,
                self.num_tokens,
                self.num_heads,
                self.out_channels,
                1,
                1
            )
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "MHPermutMixMatrix-{:?}x{:?}x{:?} {:?} heads",
            self.batch_size, self.num_tokens, self.embed_dim, self.num_heads
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, mh_permut) = input;
        let res = mh_permut.forward(tensor);
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
            LinearBenchmark, MHPermutMixBenchmark, MHPermutMixMatrixBenchmark, SpectreLinearBenchmark, SpectrePatchEmbeddingBenchmark, SpectreViTBenchmark
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

        print_bench_results("SpectrePatcher", &results, "embed_dim");
    }

    #[test]
    fn mh_permute_mix_bench() {
        type B = burn::backend::cuda::Cuda;
        type MyAutodiffBackend = Autodiff<B>;

        type CpuB = burn::backend::ndarray::NdArray;
        type CpuAutodiffBackend = Autodiff<CpuB>;

        let device = burn::backend::cuda::CudaDevice::default();
        let cpu_device = burn::backend::ndarray::NdArrayDevice::default();

        let batches = [64; 5];
        let embed_dim = [64, 128, 256, 512, 1024];
        let num_tokens = [64; 5];
        let num_heads = [1, 2, 3, 4, 6, 8];
        let out_channels = [192, 128, 256, 512, 1024];

        // GPU tests
        let mut results_gpu: Vec<(u32, BenchmarkComputations)> = Vec::new();
        let mut results_matrix_gpu: Vec<(u32, BenchmarkComputations)> = Vec::new();
        for head in num_heads.into_iter() {
            let bench = MHPermutMixBenchmark::<MyAutodiffBackend> {
                batch_size: batches[0],
                embed_dim: embed_dim[0],
                num_heads: head,
                num_tokens: num_tokens[0],
                out_channels: out_channels[0],
                device: device.clone(),
            };
            let bench_matrix = MHPermutMixMatrixBenchmark::<MyAutodiffBackend> {
                batch_size: batches[0],
                embed_dim: embed_dim[0],
                num_heads: head,
                num_tokens: num_tokens[0],
                out_channels: out_channels[0],
                device: device.clone(),
            };

            let bench_res = bench.run(TimingMethod::Device).unwrap();
            let computed = BenchmarkComputations::new(&bench_res);

            let bench_res_matrix = bench_matrix.run(TimingMethod::Device).unwrap();
            let computed_matrix = BenchmarkComputations::new(&bench_res_matrix);
            results_gpu.push((head as u32, computed));
            results_matrix_gpu.push((head as u32, computed_matrix));
        }

        // CPU tests
        let mut results_cpu: Vec<(u32, BenchmarkComputations)> = Vec::new();
        let mut results_matrix_cpu: Vec<(u32, BenchmarkComputations)> = Vec::new();
        for head in num_heads.into_iter() {
            let bench = MHPermutMixBenchmark::<CpuAutodiffBackend> {
                batch_size: batches[0],
                embed_dim: embed_dim[0],
                num_heads: head,
                num_tokens: num_tokens[0],
                out_channels: out_channels[0],
                device: cpu_device,
            };
            let bench_matrix = MHPermutMixMatrixBenchmark::<CpuAutodiffBackend> {
                batch_size: batches[0],
                embed_dim: embed_dim[0],
                num_heads: head,
                num_tokens: num_tokens[0],
                out_channels: out_channels[0],
                device: cpu_device,
            };

            let bench_res = bench.run(TimingMethod::Device).unwrap();
            let computed = BenchmarkComputations::new(&bench_res);

            let bench_res_matrix = bench_matrix.run(TimingMethod::Device).unwrap();
            let computed_matrix = BenchmarkComputations::new(&bench_res_matrix);
            results_cpu.push((head as u32, computed));
            results_matrix_cpu.push((head as u32, computed_matrix));
        }

        print_bench_results("MHPermutMix (GPU)", &results_gpu, "num_heads");
        print_bench_results("MHPermutMixMatrix (GPU)", &results_matrix_gpu, "num_heads");
        print_bench_results("MHPermutMix (NdArray)", &results_cpu, "num_heads");
        print_bench_results(
            "MHPermutMixMatrix (NdArray)",
            &results_matrix_cpu,
            "num_heads",
        );
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

        print_bench_results("SpectreViT", &results, "num_heads");
    }

    #[test]
    fn linear_bench() {
        type B = burn::backend::cuda::Cuda;
        type MyAutodiffBackend = Autodiff<B>;

        type CpuB = burn::backend::ndarray::NdArray;
        type CpuAutodiffBackend = Autodiff<CpuB>;

        let device = burn::backend::cuda::CudaDevice::default();
        let cpu_device = burn::backend::ndarray::NdArrayDevice::default();

        let batches = [64; 5];
        let embed_dim = [64, 128, 256, 512, 1024];
        let num_tokens = [64; 5];
        let out_channels = [192, 128, 256, 512, 1024];

        // GPU tests
        let mut results_gpu: Vec<(u32, BenchmarkComputations)> = Vec::new();
        let mut results_spectre_gpu: Vec<(u32, BenchmarkComputations)> = Vec::new();
        for out in out_channels.into_iter() {
            let bench_spectre_linear = SpectreLinearBenchmark::<MyAutodiffBackend> {
                batch_size: batches[0],
                num_tokens: num_tokens[0],
                in_channels: embed_dim[0],
                out_channels: out,
                device: device.clone(),
            };

            let bench_linear = LinearBenchmark::<MyAutodiffBackend> {
                batch_size: batches[0],
                num_tokens: num_tokens[0],
                in_channels: embed_dim[0],
                out_channels: out,
                device: device.clone(),
            };

            let bench_linear_res = bench_linear.run(TimingMethod::Device).unwrap();
            let computed = BenchmarkComputations::new(&bench_linear_res);

            let bench_spectre_linear_res = bench_spectre_linear.run(TimingMethod::Device).unwrap();
            let computed_spectre = BenchmarkComputations::new(&bench_spectre_linear_res);

            results_gpu.push((out as u32, computed));
            results_spectre_gpu.push((out as u32, computed_spectre));
        }

        // CPU tests
        let mut results_cpu: Vec<(u32, BenchmarkComputations)> = Vec::new();
        let mut results_spectre_cpu: Vec<(u32, BenchmarkComputations)> = Vec::new();
        for out in out_channels.into_iter() {
            let bench_spectre_linear = SpectreLinearBenchmark::<CpuAutodiffBackend> {
                batch_size: batches[0],
                num_tokens: num_tokens[0],
                in_channels: embed_dim[0],
                out_channels: out,
                device: cpu_device,
            };

            let bench_linear = LinearBenchmark::<CpuAutodiffBackend> {
                batch_size: batches[0],
                num_tokens: num_tokens[0],
                in_channels: embed_dim[0],
                out_channels: out,
                device: cpu_device,
            };

            let bench_linear_res = bench_linear.run(TimingMethod::Device).unwrap();
            let computed = BenchmarkComputations::new(&bench_linear_res);

            let bench_spectre_linear_res = bench_spectre_linear.run(TimingMethod::Device).unwrap();
            let computed_spectre = BenchmarkComputations::new(&bench_spectre_linear_res);

            results_cpu.push((out as u32, computed));
            results_spectre_cpu.push((out as u32, computed_spectre));
        }

        print_bench_results("SpectreLinear (GPU)", &results_spectre_gpu, "out_channels");
        print_bench_results("Linear (GPU)", &results_gpu, "out_channels");
        print_bench_results("SpectreLinear (NdArray)", &results_spectre_cpu, "out_channels");
        print_bench_results(
            "Linear (NdArray)",
            &results_cpu,
            "out_channels",
        );
    }


}
