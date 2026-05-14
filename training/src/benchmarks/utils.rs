use std::{fs::OpenOptions, io::Write};

use cubecl::benchmark::BenchmarkComputations;

pub fn print_bench_results(title: &str, results: &[(u32, BenchmarkComputations)], field: &str) {
    let benchmark_md = "benchmark.md";
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(benchmark_md)
        .unwrap();

    let mut buffer = String::new();
    buffer.push_str(format!("## {}\n\n", title).as_str());
    //buffer.push_str(format!("| {:-<1$} |\n", "", 85).as_str());
    buffer.push_str(
        format!(
            "| {:>10} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} |\n",
            field, "mean (µs)", "median (µs)", "variance (ns)", "min (µs)", "max (µs)"
        )
        .as_str(),
    );
    buffer.push_str(
        format!(
            "|{:-^12}|{:-^14}|{:-^14}|{:-^14}|{:-^14}|{:-^14}|\n",
            "", "", "", "", "", ""
        )
        .as_str(),
    );
    for (heads, c) in results {
        buffer.push_str(
            format!(
                "| {:>10} | {:>12.2} | {:>12.2} | {:>12.2} | {:>12.2} | {:>12.2} |\n",
                heads,
                c.mean.as_micros(),
                c.median.as_micros(),
                c.variance.as_nanos(),
                c.min.as_micros(),
                c.max.as_micros(),
            )
            .as_str(),
        );
    }
    //buffer.push_str(format!("| {:-<1$} |\n", "", 85).as_str());
    file.write_all(buffer.as_bytes()).unwrap();
    println!("{}\n\n", buffer);
}
