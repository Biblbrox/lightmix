use burn::{Tensor, prelude::Backend, tensor::Distribution};
use cubecl::benchmark::Benchmark;

use crate::mixing::{
    bandedmixer::{BandedMixer, BandedMixerConfig},
    butterflymixer::{ButterflyMixer, ButterflyMixerConfig},
    learnedmixer::{LearnedPermuter, LearnedPermuterConfig},
    randommixer::{PermutationStrategy, StaticPermuter, StaticPermuterConfig},
};

pub struct BandedMixerBenchmark<B: Backend> {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub out_channels: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for BandedMixerBenchmark<B> {
    type Input = (Tensor<B, 3>, BandedMixer<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            BandedMixerConfig::new(
                self.embed_dim,
                self.num_tokens,
                self.num_heads,
                self.out_channels,
                3,
                0.01,
            )
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "BandedMixer-{:?}x{:?}x{:?} {:?} heads",
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

pub struct ButterflyPermutBenchmark<B: Backend> {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub out_channels: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for ButterflyPermutBenchmark<B> {
    type Input = (Tensor<B, 3>, ButterflyMixer<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            ButterflyMixerConfig::new(
                self.embed_dim,
                self.num_tokens,
                self.num_heads,
                self.out_channels,
                1,
                0,
            )
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "ButterflyMixer-{:?}x{:?}x{:?} {:?} heads",
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

pub struct LearnedPermutBenchmark<B: Backend> {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub out_channels: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for LearnedPermutBenchmark<B> {
    type Input = (Tensor<B, 3>, LearnedPermuter<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            LearnedPermuterConfig::new(
                self.embed_dim,
                self.num_tokens,
                self.num_heads,
                self.out_channels,
                0.05,
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

pub struct StaticPermuterBenchmark<B: Backend> {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub out_channels: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for StaticPermuterBenchmark<B> {
    type Input = (Tensor<B, 3>, StaticPermuter<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            StaticPermuterConfig::new(
                self.embed_dim,
                self.num_tokens,
                self.num_heads,
                self.out_channels,
                1,
                PermutationStrategy::Random,
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
    use cubecl::{
        benchmark::{Benchmark, BenchmarkComputations},
        profile::TimingMethod,
    };

    use crate::{
        benchmarks::{
            CpuAutodiffBackend, CpuDevice, GpuAutodiffBackend, GpuDevice,
            mixing::{
                BandedMixerBenchmark, ButterflyPermutBenchmark, LearnedPermutBenchmark,
                StaticPermuterBenchmark,
            },
        },
        utils::print_bench_results,
    };

    #[test]
    fn mixing_benchmark() {
        let device = GpuDevice::default();
        let cpu_device = CpuDevice::default();

        let batches = [64; 5];
        let embed_dim = [64, 128, 256, 512, 1024];
        let num_tokens = [64; 5];
        let num_heads = [1, 2, 4, 8];
        let out_channels = [192, 128, 256, 512, 1024];

        // GPU tests
        let mut results_gpu_static: Vec<(u32, BenchmarkComputations)> = Vec::new();
        let mut results_gpu_learned: Vec<(u32, BenchmarkComputations)> = Vec::new();
        let mut results_gpu_butterfly: Vec<(u32, BenchmarkComputations)> = Vec::new();
        let mut results_gpu_banded: Vec<(u32, BenchmarkComputations)> = Vec::new();
        for head in num_heads.into_iter() {
            let bench_static = StaticPermuterBenchmark::<GpuAutodiffBackend> {
                batch_size: batches[0],
                embed_dim: embed_dim[0],
                num_heads: head,
                num_tokens: num_tokens[0],
                out_channels: out_channels[0],
                device: device.clone(),
            };
            let bench_learned = LearnedPermutBenchmark::<GpuAutodiffBackend> {
                batch_size: batches[0],
                embed_dim: embed_dim[0],
                num_heads: head,
                num_tokens: num_tokens[0],
                out_channels: out_channels[0],
                device: device.clone(),
            };
            let bench_butterfly = ButterflyPermutBenchmark::<GpuAutodiffBackend> {
                batch_size: batches[0],
                embed_dim: embed_dim[0],
                num_heads: head,
                num_tokens: num_tokens[0],
                out_channels: out_channels[0],
                device: device.clone(),
            };
            let bench_banded = BandedMixerBenchmark::<GpuAutodiffBackend> {
                batch_size: batches[0],
                embed_dim: embed_dim[0],
                num_heads: head,
                num_tokens: num_tokens[0],
                out_channels: out_channels[0],
                device: device.clone(),
            };

            let bench_res_static = bench_static.run(TimingMethod::Device).unwrap();
            let computed_static = BenchmarkComputations::new(&bench_res_static);

            let bench_res_learned = bench_learned.run(TimingMethod::Device).unwrap();
            let computed_learned = BenchmarkComputations::new(&bench_res_learned);

            let bench_res_butterfly = bench_butterfly.run(TimingMethod::Device).unwrap();
            let computed_butterfly = BenchmarkComputations::new(&bench_res_butterfly);

            let bench_res_banded = bench_banded.run(TimingMethod::Device).unwrap();
            let computed_banded = BenchmarkComputations::new(&bench_res_banded);

            results_gpu_static.push((head as u32, computed_static));
            results_gpu_learned.push((head as u32, computed_learned));
            results_gpu_butterfly.push((head as u32, computed_butterfly));
            results_gpu_banded.push((head as u32, computed_banded));
        }

        // CPU tests
        let mut results_cpu_static: Vec<(u32, BenchmarkComputations)> = Vec::new();
        let mut results_cpu_learned: Vec<(u32, BenchmarkComputations)> = Vec::new();
        let mut results_cpu_butterfly: Vec<(u32, BenchmarkComputations)> = Vec::new();
        let mut results_cpu_banded: Vec<(u32, BenchmarkComputations)> = Vec::new();
        for head in num_heads.into_iter() {
            let bench_learned = LearnedPermutBenchmark::<CpuAutodiffBackend> {
                embed_dim: embed_dim[0],
                num_tokens: num_tokens[0],
                batch_size: batches[0],
                num_heads: head,
                out_channels: out_channels[0],
                device: cpu_device,
            };
            let bench_static = StaticPermuterBenchmark::<CpuAutodiffBackend> {
                embed_dim: embed_dim[0],
                num_tokens: num_tokens[0],
                batch_size: batches[0],
                num_heads: head,
                out_channels: out_channels[0],
                device: cpu_device,
            };
            let bench_butterfly = ButterflyPermutBenchmark::<CpuAutodiffBackend> {
                embed_dim: embed_dim[0],
                num_tokens: num_tokens[0],
                batch_size: batches[0],
                num_heads: head,
                out_channels: out_channels[0],
                device: cpu_device,
            };
            let bench_banded = BandedMixerBenchmark::<CpuAutodiffBackend> {
                batch_size: batches[0],
                embed_dim: embed_dim[0],
                num_heads: head,
                num_tokens: num_tokens[0],
                out_channels: out_channels[0],
                device: cpu_device,
            };

            let bench_res_static = bench_static.run(TimingMethod::Device).unwrap();
            let computed_static = BenchmarkComputations::new(&bench_res_static);

            let bench_res_learned = bench_learned.run(TimingMethod::Device).unwrap();
            let computed_learned = BenchmarkComputations::new(&bench_res_learned);

            let bench_res_butterfly = bench_butterfly.run(TimingMethod::Device).unwrap();
            let computed_butterfly = BenchmarkComputations::new(&bench_res_butterfly);

            let bench_res_banded = bench_banded.run(TimingMethod::Device).unwrap();
            let computed_banded = BenchmarkComputations::new(&bench_res_banded);

            results_cpu_static.push((head as u32, computed_static));
            results_cpu_learned.push((head as u32, computed_learned));
            results_cpu_butterfly.push((head as u32, computed_butterfly));
            results_cpu_banded.push((head as u32, computed_banded));
        }

        print_bench_results("StaticPermut (GPU)", &results_gpu_static, "num_heads");
        print_bench_results("LearnedPermut (GPU)", &results_gpu_learned, "num_heads");
        print_bench_results("ButterflyPermut (GPU)", &results_gpu_butterfly, "num_heads");
        print_bench_results("BandedPermut (GPU)", &results_gpu_banded, "num_heads");
        print_bench_results("StaticPermut (NdArray)", &results_cpu_static, "num_heads");
        print_bench_results("LearnedPermut (NdArray)", &results_cpu_learned, "num_heads");
        print_bench_results(
            "ButterflyPermut (NdArray)",
            &results_cpu_butterfly,
            "num_heads",
        );
        print_bench_results("BandedPermut (NdArray)", &results_cpu_banded, "num_heads");
    }
}
