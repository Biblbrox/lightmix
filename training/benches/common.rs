use std::fs::OpenOptions;

use burn::backend::Autodiff;
use cubecl::benchmark::BenchmarkComputations;
use lightmix::attention::{
    AttentionConfig,
    csp_attention::CspConfig,
    learnedmixer::LearnedPermuterConfig,
    self_attention::SelfAttentionConfig,
    sinkformer::SinkformerMixerConfig,
    staticmixer::{PermutationStrategy, StaticMixerConfig},
    stochasticmixer::StochasticMixerConfig,
    stochasticwindowmixer::StochasticWindowMixerConfig,
};
use serde::Serialize;
use std::io::Write;

/// Simple enum listing attention variants for benchmark iteration.
#[derive(Debug, Clone, Copy)]
pub enum AttentionVariant {
    SelfAttention,
    Sinkformer,
    Csp,
    StochasticMixer,
    StaticMixer,
    LearnedPermuter,
    StochasticWindow,
}

impl AttentionVariant {
    pub fn all_variants() -> &'static [AttentionVariant] {
        &[
            Self::SelfAttention,
            Self::Sinkformer,
            Self::Csp,
            Self::StochasticMixer,
            Self::StaticMixer,
            Self::LearnedPermuter,
            Self::StochasticWindow,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::SelfAttention => "SelfAttention",
            Self::Sinkformer => "Sinkformer",
            Self::Csp => "CSP",
            Self::StochasticMixer => "StochasticMixer",
            Self::StaticMixer => "StaticMixer",
            Self::LearnedPermuter => "LearnedPermuter",
            Self::StochasticWindow => "StochasticWindow",
        }
    }
}

/// Construct an `AttentionConfig` from common benchmark parameters.
/// Variant-specific fields use sensible defaults.
pub fn make_attention_config(
    variant: AttentionVariant,
    embed_dim: usize,
    num_heads: usize,
    seq_length: usize,
) -> AttentionConfig {
    match variant {
        AttentionVariant::SelfAttention => {
            AttentionConfig::SelfAttention(SelfAttentionConfig::new(embed_dim, num_heads))
        }
        AttentionVariant::Sinkformer => {
            AttentionConfig::Sinkformer(SinkformerMixerConfig::new(
                embed_dim,
                num_heads,
                3,
                0.05,
            ))
        }
        AttentionVariant::Csp => {
            AttentionConfig::Csp(CspConfig::new(embed_dim, seq_length, (seq_length / 2).max(1)))
        }
        AttentionVariant::StochasticMixer => {
            AttentionConfig::StochasticMixer(StochasticMixerConfig::new(embed_dim, num_heads, 0.05))
        }
        AttentionVariant::StaticMixer => {
            AttentionConfig::StaticMixer(StaticMixerConfig::new(
                embed_dim,
                seq_length,
                num_heads,
                PermutationStrategy::Random,
            ))
        }
        AttentionVariant::LearnedPermuter => {
            AttentionConfig::LearnedPermuter(LearnedPermuterConfig::new(
                embed_dim,
                seq_length,
                0,
                0.05,
            ))
        }
        AttentionVariant::StochasticWindow => {
            AttentionConfig::StochasticWindow(StochasticWindowMixerConfig::new(
                embed_dim,
                seq_length,
                num_heads,
                3,
                0.01,
            ))
        }
    }
}

pub type GpuBackend = burn::backend::cuda::Cuda;
pub type GpuAutodiffBackend = Autodiff<GpuBackend>;
pub type GpuDevice = burn::backend::cuda::CudaDevice;

pub type CpuBackend = burn::backend::flex::Flex;
pub type CpuAutodiffBackend = Autodiff<CpuBackend>;
pub type CpuDevice = burn::backend::flex::FlexDevice;

#[derive(Serialize)]
pub struct BenchmarkRun {
    pub run_id: String,
    pub bench_file: String,
    pub backend: String,
    pub title: String,
    pub row_field: String,
    pub rows: Vec<BenchmarkRow>,
}

#[derive(Serialize)]
pub struct BenchmarkRow {
    pub field_value: u32,
    pub mean_us: f64,
    pub median_us: f64,
    pub variance_ns: f64,
    pub min_us: f64,
    pub max_us: f64,
}

pub fn run_to_row(computed: &BenchmarkComputations, field_value: u32) -> BenchmarkRow {
    BenchmarkRow {
        field_value,
        mean_us: computed.mean.as_micros() as f64,
        median_us: computed.median.as_micros() as f64,
        variance_ns: computed.variance.as_nanos() as f64,
        min_us: computed.min.as_micros() as f64,
        max_us: computed.max.as_micros() as f64,
    }
}

pub fn print_bench_results(
    run_id: &str,
    bench_file: &str,
    backend: &str,
    title: &str,
    row_field: &str,
    results: &[(u32, BenchmarkComputations)],
) -> String {
    let rows = results
        .iter()
        .map(|(value, computed)| run_to_row(computed, *value))
        .collect::<Vec<_>>();

    let run = BenchmarkRun {
        run_id: run_id.to_string(),
        bench_file: bench_file.to_string(),
        backend: backend.to_string(),
        title: title.to_string(),
        row_field: row_field.to_string(),
        rows,
    };

    let json = serde_json::to_string(&run).unwrap();

    let path = benchmark_output_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(file, "{}", json);
    }

    json
}

pub fn generate_run_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    if let Ok(run_id) = std::env::var("LIGHTMIX_RUN_ID") {
        return run_id;
    }
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = duration.as_secs();
    let nanos = duration.subsec_nanos();
    format!("run_{secs}_{nanos}")
}

fn benchmark_output_path() -> std::path::PathBuf {
    use std::env;
    env::var("LIGHTMIX_BENCH_OUTPUT")
        .ok()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let mut p = env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
            p.push("benchmarks");
            p.push("results.jsonl");
            p
        })
}
