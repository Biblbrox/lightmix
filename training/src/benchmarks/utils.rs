use std::fs::OpenOptions;
use std::io::Write;

use cubecl::benchmark::BenchmarkComputations;

use super::results::BenchmarkRun;

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
        .map(|(value, computed)| super::results::run_to_row(computed, *value))
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
