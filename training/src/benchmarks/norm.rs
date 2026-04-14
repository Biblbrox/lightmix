use burn::{
    Tensor,
    nn::{LayerNorm, LayerNormConfig},
    prelude::Backend,
    tensor::Distribution,
};
use cubecl::benchmark::Benchmark;

use crate::norm::{DynamicERF, DynamicERFConfig};

pub struct DerfBenchmark<B: Backend> {
    batch_size: usize,
    num_tokens: usize,
    embed_dim: usize,
    normalized_shape: usize,
    alpha_init_value: f32,
    shift_init_value: f32,
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
            DynamicERFConfig::new(
                self.normalized_shape,
                self.alpha_init_value,
                self.shift_init_value,
            )
            .init(&self.device),
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

#[cfg(test)]
mod tests {
    use cubecl::{
        benchmark::{Benchmark, BenchmarkComputations},
        profile::TimingMethod,
    };

    use crate::{
        benchmarks::norm::{DerfBenchmark, LayerNormBenchmark},
        utils::print_bench_results,
    };

    #[test]
    fn norm_bench() {
        type B = burn::backend::cuda::Cuda;
        let device = burn::backend::cuda::CudaDevice::default();

        let batches = [8; 5];
        let embed_dim = [64, 128, 256, 512, 1024];
        let num_tokens = [65; 5];

        let mut erf_results: Vec<(u32, BenchmarkComputations)> = Vec::new();
        let mut layer_norm_results: Vec<(u32, BenchmarkComputations)> = Vec::new();
        for embed in embed_dim.into_iter() {
            let erf_bench = DerfBenchmark::<B> {
                normalized_shape: embed,
                alpha_init_value: 0.5,
                shift_init_value: 0.0,
                batch_size: batches[0],
                embed_dim: embed,
                num_tokens: num_tokens[0],
                device: device.clone(),
            };

            let bench_res = erf_bench.run(TimingMethod::Device).unwrap();
            let computed = BenchmarkComputations::new(&bench_res);
            erf_results.push((embed as u32, computed));

            let layer_norm_bench = LayerNormBenchmark::<B> {
                normalized_shape: embed,
                batch_size: batches[0],
                embed_dim: embed,
                num_tokens: num_tokens[0],
                device: device.clone(),
            };

            let bench_res = layer_norm_bench.run(TimingMethod::Device).unwrap();
            let computed = BenchmarkComputations::new(&bench_res);
            layer_norm_results.push((embed as u32, computed));
        }

        print_bench_results("ERF", &erf_results, "embed_dim");
        print_bench_results("LayerNorm", &layer_norm_results, "embed_dim");
    }
}
