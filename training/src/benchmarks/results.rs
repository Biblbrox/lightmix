use cubecl::benchmark::BenchmarkComputations;
use serde::Serialize;

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
