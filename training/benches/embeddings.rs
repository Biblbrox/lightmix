use burn::{
    Tensor,
    tensor::{Distribution, backend::Backend},
};
use cubecl::benchmark::Benchmark;

use lightmix::embeddings::vit::{PatchEmbedding, PatchEmbeddingConfig};

pub struct PatchEmbeddingBenchmark<B: Backend> {
    pub batch_size: usize,
    pub in_channels: usize,
    pub embed_dim: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub seq_length: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for PatchEmbeddingBenchmark<B> {
    type Input = (Tensor<B, 4>, PatchEmbedding<B>);
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
            PatchEmbeddingConfig::new(
                self.in_channels,
                self.embed_dim,
                self.patch_size,
                self.image_size,
                0.1,
                self.seq_length,
                true,
            )
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "PatchEmbeddingBenchmark-{:?}x{:?}x{:?}x{:?} image",
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

fn main() {
    use cubecl::{
        benchmark::{Benchmark, BenchmarkComputations},
        profile::TimingMethod,
    };

    use lightmix::benchmarks::{GpuBackend, GpuDevice, utils::{print_bench_results, generate_run_id}};

    let device = GpuDevice::default();
    let run_id = generate_run_id();

    let batches = [8; 5];
    let embed_dim = [64, 128, 256, 512, 1024];
    let image_size: usize = 224;
    let patch_size: usize = 16;
    let in_channels = 3;

    let mut results: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let num_patches = (image_size / patch_size).pow(2);
    for embed in embed_dim.into_iter() {
        let bench = PatchEmbeddingBenchmark::<GpuBackend> {
            batch_size: batches[0],
            in_channels,
            embed_dim: embed,
            patch_size,
            image_size,
            seq_length: num_patches,
            device: device.clone(),
        };

        let bench_res = bench.run(TimingMethod::System).unwrap();
        let computed = BenchmarkComputations::new(&bench_res);
        results.push((embed as u32, computed));
    }

    print_bench_results(
        &run_id, "embeddings", "GPU",
        "Patcher",
        "embed_dim",
        &results,
    );
}
