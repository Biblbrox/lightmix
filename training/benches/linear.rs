use burn::tensor::backend::Backend;
use burn::{Tensor, tensor::Distribution};
use cubecl::benchmark::Benchmark;
use cubecl::{benchmark::BenchmarkComputations, profile::TimingMethod};

use burn::nn::{Linear, LinearConfig};
use embed_former_train::linear::monarch::{MonarchLinear, MonarchLinearConfig};
use embed_former_train::{
    benchmarks::utils::print_bench_results,
    benchmarks::{CpuBackend, GpuBackend},
};

// ── MonarchLinear benchmark ──────────────────────────────────────────────────
pub struct MonarchLinearBenchmark<B: Backend> {
    pub batch_size: usize,
    pub seq_len: usize,
    pub embed_dim: usize, // must be a perfect square
    pub device: B::Device,
    pub model: MonarchLinear<B>,
}

impl<B: Backend> Benchmark for MonarchLinearBenchmark<B> {
    type Input = Tensor<B, 3>;
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        Tensor::<B, 3>::random(
            [self.batch_size, self.seq_len, self.embed_dim],
            Distribution::Default,
            &self.device,
        )
    }

    fn name(&self) -> String {
        format!(
            "monarch-linear-{:?}x{:?}x{:?}",
            self.batch_size, self.seq_len, self.embed_dim,
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        Ok(self.model.forward(input))
    }
}

// ── Standard Linear benchmark (baseline) ────────────────────────────────────
pub struct LinearBenchmark<B: Backend> {
    pub batch_size: usize,
    pub seq_len: usize,
    pub embed_dim: usize,
    pub device: B::Device,
    pub model: Linear<B>,
}

impl<B: Backend> Benchmark for LinearBenchmark<B> {
    type Input = Tensor<B, 3>;
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        Tensor::<B, 3>::random(
            [self.batch_size, self.seq_len, self.embed_dim],
            Distribution::Default,
            &self.device,
        )
    }

    fn name(&self) -> String {
        format!(
            "linear-{:?}x{:?}x{:?}",
            self.batch_size, self.seq_len, self.embed_dim,
        )
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        Ok(self.model.forward(input))
    }
}

fn monarch_benchmark_backend<B: Backend>(backend: &str) {
    let device = B::Device::default();

    let batch_size = 64;
    let seq_len = 196;
    // Must be perfect squares: 64=8^2, 256=16^2, and so on
    let embed_dims = [64usize, 256, 576, 1024, 4096];

    let mut results_monarch: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut results_linear: Vec<(u32, BenchmarkComputations)> = Vec::new();

    for &dim in embed_dims.iter() {
        let bench_monarch = MonarchLinearBenchmark::<B> {
            batch_size,
            seq_len,
            embed_dim: dim,
            device: device.clone(),
            model: MonarchLinearConfig::new(dim, dim)
                .with_bias(false)
                .init(&device),
        };

        let bench_linear = LinearBenchmark::<B> {
            batch_size,
            seq_len,
            embed_dim: dim,
            device: device.clone(),
            model: LinearConfig::new(dim, dim).with_bias(false).init(&device),
        };

        let res_monarch = bench_monarch.run(TimingMethod::System).unwrap();
        let res_linear = bench_linear.run(TimingMethod::System).unwrap();

        results_monarch.push((dim as u32, BenchmarkComputations::new(&res_monarch)));
        results_linear.push((dim as u32, BenchmarkComputations::new(&res_linear)));
    }

    print_bench_results(
        format!("MonarchLinear ({})", backend).as_str(),
        &results_monarch,
        "embed_dim",
    );
    print_bench_results(
        format!("Linear baseline ({})", backend).as_str(),
        &results_linear,
        "embed_dim",
    );
}

fn main() {
    println!("=== MonarchLinear vs Linear Benchmarks ===");
    monarch_benchmark_backend::<GpuBackend>("GPU");
    monarch_benchmark_backend::<CpuBackend>("CPU");
}
