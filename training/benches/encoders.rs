use burn::{
    Tensor,
    nn::transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    tensor::{Distribution, backend::Backend},
};
use cubecl::benchmark::Benchmark;

use lightmix::{
    benchmarks::{CpuBackend, GpuBackend, utils::print_bench_results},
    encoders::fast_encoder::{FastEncoder, FastEncoderConfig},
};

use cubecl::{benchmark::BenchmarkComputations, profile::TimingMethod};

pub struct FastEncoderBenchmark<B: Backend> {
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

impl<B: Backend> Benchmark for FastEncoderBenchmark<B> {
    type Input = (Tensor<B, 3>, FastEncoder<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        let grid_size = self.image_size / self.patch_size;
        let num_patches = grid_size.pow(2);
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_patches, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            FastEncoderConfig::new(
                self.num_layers,
                grid_size,
                num_patches,
                self.embed_dim,
                self.num_heads,
                self.hid_dim,
                self.dropout,
                0.05,
            )
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "FastEncoder-{:?}x{:?}x{:?} {:?} heads",
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

pub struct ViTEncoderBenchmark<B: Backend> {
    pub num_patches: usize,
    pub batch_size: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub hid_dim: usize,
    pub dropout: f64,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for ViTEncoderBenchmark<B> {
    type Input = (Tensor<B, 3>, TransformerEncoder<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_patches, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            TransformerEncoderConfig::new(
                self.embed_dim,
                self.hid_dim,
                self.num_heads,
                self.num_layers,
            )
            .with_dropout(self.dropout)
            .with_norm_first(true)
            .with_activation(burn::nn::activation::ActivationConfig::Gelu)
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "ViTEncoder-{:?}x{:?}x{:?} {:?} heads",
            self.batch_size, self.num_patches, self.embed_dim, self.num_heads
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, model) = input;
        let res = model.forward(TransformerEncoderInput::new(tensor));
        Ok(res)
    }
}

fn encoders_benchmark_backend<B: Backend>(run_id: &str, backend: &str) {
    let device = B::Device::default();

    let batches = [8; 5];
    let embed_dim = [64, 128, 256, 512, 1024];
    let num_heads = [1, 2, 4, 8, 16];
    let _out_channels = [64, 128, 256, 512, 1024];
    let image_size: usize = 224;
    let patch_size: usize = 16;
    let num_patches: usize = (image_size / patch_size).pow(2);
    let num_classes = 1000;
    let num_layers = 12;
    let hid_dim = 768;
    let in_channels = 3;
    let dropout = 0.5;

    let mut results_fast: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut results_vit: Vec<(u32, BenchmarkComputations)> = Vec::new();
    for head in num_heads.into_iter() {
        let bench_fast = FastEncoderBenchmark::<B> {
            num_patches,
            batch_size: batches[0],
            embed_dim: embed_dim[0],
            num_heads: head,
            num_layers,
            in_channels,
            num_classes,
            patch_size,
            image_size,
            hid_dim,
            device: device.clone(),
            dropout,
        };

        let bench_vit = ViTEncoderBenchmark::<B> {
            num_patches,
            batch_size: batches[0],
            embed_dim: embed_dim[0],
            num_heads: head,
            num_layers,
            hid_dim,
            device: device.clone(),
            dropout,
        };

        let bench_res_fast = bench_fast.run(TimingMethod::System).unwrap();
        let bench_res_vit = bench_vit.run(TimingMethod::System).unwrap();
        let computed_fast = BenchmarkComputations::new(&bench_res_fast);
        let computed_vit = BenchmarkComputations::new(&bench_res_vit);
        results_fast.push((head as u32, computed_fast));
        results_vit.push((head as u32, computed_vit));
    }

    print_bench_results(
        run_id, "encoders", backend,
        &format!("FastEncoder ({})", backend),
        "num_heads",
        &results_fast,
    );
    print_bench_results(
        run_id, "encoders", backend,
        &format!("ViTEncoder ({})", backend),
        "num_heads",
        &results_vit,
    );
}

fn main() {
    use lightmix::benchmarks::utils::generate_run_id;

    let run_id = generate_run_id();
    encoders_benchmark_backend::<GpuBackend>(&run_id, "GPU");
    encoders_benchmark_backend::<CpuBackend>(&run_id, "CPU");
}
