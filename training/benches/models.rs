use burn::{
    Tensor,
    tensor::{Distribution, backend::Backend},
};
use cubecl::benchmark::Benchmark;

use lightmix::{
    benchmarks::utils::generate_run_id,
    models::{
        efficientvit::{EfficientViT, EfficientViTConfig},
        fast_vit::{FastViT, FastViTConfig},
        vit::{ViT, ViTConfig},
    },
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
            FastViTConfig {
                embed_dim: self.embed_dim,
                num_heads: self.num_heads,
                num_encoders: self.num_layers,
                patch_size: self.patch_size,
                hidden_dim: self.hid_dim,
                sinkhorn_temp: 0.05,
                activation: "gelu".to_string(),
                dropout: self.dropout,
            }
            .init(
                &self.device,
                self.in_channels,
                self.image_size,
                self.num_classes,
            ),
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
            ViTConfig {
                embed_dim: self.embed_dim,
                num_heads: self.num_heads,
                num_encoders: self.num_layers,
                patch_size: self.patch_size,
                hidden_dim: self.hid_dim,
                dropout: self.dropout,
            }
            .init(
                &self.device,
                self.in_channels,
                self.image_size,
                self.num_classes,
            ),
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

pub struct EfficientViTBenchmark<B: Backend> {
    pub num_patches: usize,
    pub batch_size: usize,
    pub in_channels: usize,
    pub image_size: usize,
    pub num_classes: usize,
    pub config: EfficientViTConfig,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for EfficientViTBenchmark<B> {
    type Input = (Tensor<B, 4>, EfficientViT<B>);
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
            self.config.clone().init(
                &self.device,
                self.in_channels,
                self.image_size,
                self.num_classes,
            ),
        )
    }

    fn name(&self) -> String {
        let total_depth = self.config.stage_depths.iter().sum::<usize>();

        format!(
            "EfficientViT-{:?}x{:?}x{:?} depth{:?}",
            self.batch_size, self.num_patches, self.config.stem_channels, total_depth
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

fn make_efficientvit_config(
    stem_channels: usize,
    stage_channels: [usize; 3],
    stage_depths: [usize; 3],
    stage_heads: [usize; 3],
    ffn_expansion_ratio: usize,
    mbconv_expansion_ratio: usize,
    attention_kernel_size: usize,
    dropout: f64,
    adam_weight_decay: f64,
    adam_betas: [f64; 2],
) -> EfficientViTConfig {
    EfficientViTConfig {
        stem_channels,
        stage_channels,
        stage_depths,
        stage_heads,
        ffn_expansion_ratio,
        mbconv_expansion_ratio,
        attention_kernel_size,
        dropout,
        adam_weight_decay,
        adam_betas,
    }
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
    let mut results_efficientvit: Vec<(u32, BenchmarkComputations)> = Vec::new();
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

        let efficientvit_config = match i {
            0 => make_efficientvit_config(
                32,
                [64, 128, 256],
                [2, 4, 6], // total depth = 12
                [2, 4, 8],
                2,
                4,
                3,
                dropout,
                0.025,
                [0.9, 0.999],
            ),
            1 => make_efficientvit_config(
                48,
                [96, 192, 384],
                [3, 3, 6], // total depth = 12
                [3, 6, 12],
                2,
                4,
                3,
                dropout,
                0.025,
                [0.9, 0.999],
            ),
            _ => make_efficientvit_config(
                64,
                [128, 256, 512],
                [4, 4, 4], // total depth = 12
                [4, 8, 16],
                2,
                4,
                3,
                dropout,
                0.025,
                [0.9, 0.999],
            ),
        };

        let bench_efficientvit = EfficientViTBenchmark::<B> {
            num_patches,
            batch_size,
            in_channels,
            image_size,
            num_classes,
            config: efficientvit_config,
            device: device.clone(),
        };
        let bench_res_efficientvit = bench_efficientvit.run(TimingMethod::System).unwrap();
        let computed_efficientvit = BenchmarkComputations::new(&bench_res_efficientvit);

        results_efficientvit.push((i as u32, computed_efficientvit));
        results_fast.push((i as u32, computed_fast));
        results_vit.push((i as u32, computed_vit));
    }

    let run_id = generate_run_id();

    print_bench_results(
        run_id.as_str(),
        "models",
        backend,
        &format!("FastViT ({})", backend),
        "embed_dim",
        &results_fast,
    );

    print_bench_results(
        run_id.as_str(),
        "models",
        backend,
        &format!("ViT ({})", backend),
        "embed_dim",
        &results_vit,
    );

    print_bench_results(
        run_id.as_str(),
        "models",
        backend,
        &format!("EfficientViT ({})", backend),
        "variant",
        &results_efficientvit,
    );
}

fn main() {
    models_benchmark_backend::<GpuBackend>("GPU");
    //models_benchmark_backend::<GpuAutodiffBackend>("Autodiff GPU");
    //models_benchmark_backend::<CpuBackend>("CPU");
}
