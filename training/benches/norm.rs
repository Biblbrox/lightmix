use burn::{
    Tensor,
    nn::{LayerNorm, LayerNormConfig},
    tensor::{Distribution, backend::Backend},
};
use cubecl::benchmark::Benchmark;

use lightmix::norm::{DynamicERF, DynamicERFConfig};

use crate::common::{
    GpuAutodiffBackend, GpuBackend, GpuDevice, generate_run_id, print_bench_results,
};
mod common;

pub struct DerfBenchmark<B: Backend> {
    batch_size: usize,
    num_tokens: usize,
    embed_dim: usize,
    normalized_shape: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for DerfBenchmark<B> {
    type Input = (Tensor<B, 3>, DynamicERF<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            DynamicERFConfig::new(self.normalized_shape).init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "DerfBenchmark-{:?}x{:?}x{:?}",
            self.batch_size, self.num_tokens, self.embed_dim
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

pub struct LayerNormBenchmark<B: Backend> {
    batch_size: usize,
    num_tokens: usize,
    embed_dim: usize,
    normalized_shape: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for LayerNormBenchmark<B> {
    type Input = (Tensor<B, 3>, LayerNorm<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            LayerNormConfig::new(self.normalized_shape).init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "LayerNormBenchmark-{:?}x{:?}x{:?}",
            self.batch_size, self.num_tokens, self.embed_dim
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

fn main() {
    use cubecl::{
        benchmark::{Benchmark, BenchmarkComputations},
        profile::TimingMethod,
    };

    let device = GpuDevice::default();
    let run_id = generate_run_id();

    let batches = [8; 5];
    let embed_dim = [64, 128, 256, 512, 1024];
    let num_tokens = [65; 5];

    let mut erf_results: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut layer_norm_results: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut erf_results_autodiff: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut layer_norm_results_autodiff: Vec<(u32, BenchmarkComputations)> = Vec::new();
    for embed in embed_dim.into_iter() {
        // Validation results
        let erf_bench = DerfBenchmark::<GpuBackend> {
            normalized_shape: embed,
            batch_size: batches[0],
            embed_dim: embed,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };

        let bench_res = erf_bench.run(TimingMethod::Device).unwrap();
        let computed = BenchmarkComputations::new(&bench_res);
        erf_results.push((embed as u32, computed));

        // Autodiff results
        let erf_bench = DerfBenchmark::<GpuAutodiffBackend> {
            normalized_shape: embed,
            batch_size: batches[0],
            embed_dim: embed,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };

        let bench_res = erf_bench.run(TimingMethod::Device).unwrap();
        let computed = BenchmarkComputations::new(&bench_res);
        erf_results_autodiff.push((embed as u32, computed));

        // Validation results
        let layer_norm_bench = LayerNormBenchmark::<GpuBackend> {
            normalized_shape: embed,
            batch_size: batches[0],
            embed_dim: embed,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };

        let bench_res = layer_norm_bench.run(TimingMethod::Device).unwrap();
        let computed = BenchmarkComputations::new(&bench_res);
        layer_norm_results.push((embed as u32, computed));

        // Autodiff results
        let layer_norm_bench = LayerNormBenchmark::<GpuAutodiffBackend> {
            normalized_shape: embed,
            batch_size: batches[0],
            embed_dim: embed,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };

        let bench_res = layer_norm_bench.run(TimingMethod::Device).unwrap();
        let computed = BenchmarkComputations::new(&bench_res);
        layer_norm_results_autodiff.push((embed as u32, computed));
    }

    print_bench_results(&run_id, "norm", "GPU", "ERF", "embed_dim", &erf_results);
    print_bench_results(
        &run_id,
        "norm",
        "GPU",
        "LayerNorm",
        "embed_dim",
        &layer_norm_results,
    );
    print_bench_results(
        &run_id,
        "norm",
        "Autodiff GPU",
        "ERF (Autodiff)",
        "embed_dim",
        &erf_results_autodiff,
    );
    print_bench_results(
        &run_id,
        "norm",
        "Autodiff GPU",
        "LayerNorm (Autodiff)",
        "embed_dim",
        &layer_norm_results_autodiff,
    );
}
