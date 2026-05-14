use burn::{
    Tensor,
    nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    tensor::{Distribution, backend::Backend},
};
use cubecl::benchmark::Benchmark;

use embed_former_train::{
    benchmarks::{CpuBackend, GpuAutodiffBackend},
    mixing::{
        bandedmixer::{BandedMixer, BandedMixerConfig},
        learnedmixer::{LearnedPermuter, LearnedPermuterConfig},
        staticmixer::{PermutationStrategy, StaticMixer, StaticMixerConfig},
    },
};

// Fix the incomplete execute method in SelfAttentionBenchmark
impl<B: Backend> Benchmark for SelfAttentionBenchmark<B> {
    type Input = (Tensor<B, 3>, MultiHeadAttention<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            MultiHeadAttentionConfig::new(self.embed_dim, self.num_heads).init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "SelfAttention-{:?}x{:?}x{:?} {:?} heads",
            self.batch_size, self.num_tokens, self.embed_dim, self.num_heads
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, mha) = input;
        let res = mha.forward(MhaInput::self_attn(tensor));
        Ok(res.context)
    }
}

pub struct SelfAttentionBenchmark<B: Backend> {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub device: B::Device,
}

#[derive(Clone, Copy)]
pub enum BandedMixerMode {
    Soft,
    Hard,
}

pub struct BandedMixerBenchmark<B: Backend> {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub mode: BandedMixerMode,
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
            BandedMixerConfig::new(self.embed_dim, self.num_tokens, self.num_heads, 3, 0.01)
                .with_use_monarch(false)
                .init(&self.device),
        )
    }

    fn name(&self) -> String {
        let mode = match self.mode {
            BandedMixerMode::Soft => "soft",
            BandedMixerMode::Hard => "hard",
        };
        format!(
            "BandedMixer[{}]-{:?}x{:?}x{:?} {:?} heads",
            mode, self.batch_size, self.num_tokens, self.embed_dim, self.num_heads
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, mixer) = input;
        Ok(match self.mode {
            BandedMixerMode::Soft => mixer.forward(tensor),
            BandedMixerMode::Hard => mixer.forward_hard(tensor),
        })
    }
}

pub struct LearnedPermutBenchmark<B: Backend> {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub batch_size: usize,
    pub num_heads: usize,
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
            LearnedPermuterConfig::new(self.embed_dim, self.num_tokens, self.num_heads, 0.05)
                .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "LearnedPermuter-{:?}x{:?}x{:?} {:?} heads",
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
    pub device: B::Device,
}

impl<B: Backend> Benchmark for StaticPermuterBenchmark<B> {
    type Input = (Tensor<B, 3>, StaticMixer<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            StaticMixerConfig::new(
                self.embed_dim,
                self.num_tokens,
                self.num_heads,
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
        let res = mh_permut.forward_hard(tensor);
        Ok(res)
    }
}

fn main() {
    use embed_former_train::benchmarks::GpuBackend;

    println!("=== Mixing Benchmarks ===");
    mixing_benchmark_backend::<GpuBackend>("Gpu");
    mixing_benchmark_backend::<CpuBackend>("Cpu");
    mixing_benchmark_backend::<GpuAutodiffBackend>("Gpu (Autodiff)");
}

fn mixing_benchmark_backend<B: Backend>(backend: &str) {
    use cubecl::benchmark::BenchmarkComputations;
    use cubecl::profile::TimingMethod;
    use embed_former_train::benchmarks::utils::print_bench_results;

    let device = B::Device::default();

    let batches = [64; 6];
    let embed_dim = [64, 128, 256, 512, 1024, 2048];
    let num_tokens = [64; 6];
    let num_heads = [1, 2, 4, 8, 16];

    let mut results_attn: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut results_static: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut results_learned: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut results_banded_soft: Vec<(u32, BenchmarkComputations)> = Vec::new();
    let mut results_banded_hard: Vec<(u32, BenchmarkComputations)> = Vec::new();

    for head in num_heads.into_iter() {
        let bench_attn = SelfAttentionBenchmark::<B> {
            batch_size: batches[0],
            embed_dim: embed_dim[0],
            num_heads: head,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };

        let bench_static = StaticPermuterBenchmark::<B> {
            batch_size: batches[0],
            embed_dim: embed_dim[0],
            num_heads: head,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };
        let bench_learned = LearnedPermutBenchmark::<B> {
            batch_size: batches[0],
            embed_dim: embed_dim[0],
            num_heads: head,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };

        let bench_banded_soft = BandedMixerBenchmark::<B> {
            batch_size: batches[0],
            embed_dim: embed_dim[0],
            num_heads: head,
            mode: BandedMixerMode::Soft,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };

        let bench_banded_hard = BandedMixerBenchmark::<B> {
            batch_size: batches[0],
            embed_dim: embed_dim[0],
            num_heads: head,
            mode: BandedMixerMode::Hard,
            num_tokens: num_tokens[0],
            device: device.clone(),
        };

        let bench_res_attn = bench_attn.run(TimingMethod::System).unwrap();
        results_attn.push((head as u32, BenchmarkComputations::new(&bench_res_attn)));

        let bench_res_static = bench_static.run(TimingMethod::System).unwrap();
        results_static.push((head as u32, BenchmarkComputations::new(&bench_res_static)));

        let bench_res_learned = bench_learned.run(TimingMethod::System).unwrap();
        results_learned.push((head as u32, BenchmarkComputations::new(&bench_res_learned)));

        let bench_res_banded_soft = bench_banded_soft.run(TimingMethod::System).unwrap();
        results_banded_soft.push((
            head as u32,
            BenchmarkComputations::new(&bench_res_banded_soft),
        ));
        let bench_res_banded_hard = bench_banded_hard.run(TimingMethod::System).unwrap();
        results_banded_hard.push((
            head as u32,
            BenchmarkComputations::new(&bench_res_banded_hard),
        ));
    }

    print_bench_results(
        format!("SelfAttention/ViT ({})", backend).as_str(),
        &results_attn,
        "num_heads",
    );
    print_bench_results(
        format!("StaticPermut ({})", backend).as_str(),
        &results_static,
        "num_heads",
    );
    print_bench_results(
        format!("LearnedPermut ({})", backend).as_str(),
        &results_learned,
        "num_heads",
    );
    print_bench_results(
        format!("BandedPermut Soft ({})", backend).as_str(),
        &results_banded_soft,
        "num_heads",
    );
    print_bench_results(
        format!("BandedPermut Hard ({})", backend).as_str(),
        &results_banded_hard,
        "num_heads",
    );
}
