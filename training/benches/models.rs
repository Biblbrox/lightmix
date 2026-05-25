use burn::{
    Tensor,
    tensor::{Distribution, backend::Backend},
};
use cubecl::benchmark::Benchmark;

use lightmix::models::{
    fast_vit::{FastViT, FastViTConfig},
    vit::{ViT, ViTConfig},
};

use cubecl::{benchmark::BenchmarkComputations, profile::TimingMethod};

use lightmix::benchmarks::GpuBackend;
use lightmix::benchmarks::utils::print_bench_results;

pub struct FastViTBenchmark<B: Backend> {
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

impl<B: Backend> Benchmark for FastViTBenchmark<B> {
    type Input = (Tensor<B, 4>, FastViT<B>);
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
            FastViTConfig::new(
                self.in_channels,
                self.embed_dim,
                self.num_heads,
                self.num_layers,
                self.num_classes,
                self.patch_size,
                self.image_size,
                self.hid_dim,
                self.dropout,
                0.05,
            )
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "FastViT-{:?}x{:?}x{:?} {:?} heads",
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

pub struct ViTBenchmark<B: Backend> {
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

impl<B: Backend> Benchmark for ViTBenchmark<B> {
    type Input = (Tensor<B, 4>, ViT<B>);
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
            ViTConfig::new(
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
            "ViT-{:?}x{:?}x{:?} {:?} heads",
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

pub struct ParallelViTBenchmark<B: Backend> {
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

fn models_benchmark_backend<B: Backend>(backend: &str) {
    let device = B::Device::default();

    let batch_size = 8;
    let embed_dim = [192, 384, 768];
    let hid_dim = [768, 1536, 3072];
    let num_heads = [3, 6, 12];
    let layers = [12, 12, 12];
    let image_size: usize = 224;
    let patch_size: usize = 16;
    let num_patches: usize = (image_size / patch_size).pow(2);
    let num_classes = 100;
    let in_channels = 3;
    let dropout = 0.0;

    let mut results_fast: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut results_vit: Vec<(u32, BenchmarkComputations)> = Vec::new();
    for i in 0..num_heads.len() {
        let bench_fast = FastViTBenchmark::<B> {
            num_patches,
            batch_size: batch_size,
            embed_dim: embed_dim[i],
            num_heads: num_heads[i],
            num_layers: layers[i],
            in_channels,
            num_classes,
            patch_size,
            image_size,
            hid_dim: hid_dim[i],
            device: device.clone(),
            dropout,
        };
        let bench_res_fast = bench_fast.run(TimingMethod::System).unwrap();
        let computed_fast = BenchmarkComputations::new(&bench_res_fast);

        let bench_vit = ViTBenchmark::<B> {
            num_patches,
            batch_size: batch_size,
            embed_dim: embed_dim[i],
            num_heads: num_heads[i],
            num_layers: layers[i],
            in_channels,
            num_classes,
            patch_size,
            image_size,
            hid_dim: hid_dim[i],
            device: device.clone(),
            dropout,
        };
        let bench_res_vit = bench_vit.run(TimingMethod::System).unwrap();
        let computed_vit = BenchmarkComputations::new(&bench_res_vit);

        results_fast.push((i as u32, computed_fast));
        results_vit.push((i as u32, computed_vit));
    }

    print_bench_results(
        format!("FastViT ({})", backend).as_str(),
        &results_fast,
        "Model size",
    );

    print_bench_results(
        format!("ViT ({})", backend).as_str(),
        &results_vit,
        "Model size",
    );
}

fn main() {
    models_benchmark_backend::<GpuBackend>("GPU");
    //models_benchmark_backend::<GpuAutodiffBackend>("Autodiff GPU");
    //models_benchmark_backend::<CpuBackend>("CPU");
}
