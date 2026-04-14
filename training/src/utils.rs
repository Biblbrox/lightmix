use cubecl::benchmark::BenchmarkComputations;

pub fn print_bench_results(title: &str, results: &[(u32, BenchmarkComputations)], field: &str) {
    println!("| {:-^1$} |", title, 85);
    println!("| {:-<1$} |", "", 85);
    println!(
        "| {:>10} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} |",
        field, "mean (µs)", "median (µs)", "variance (ns)", "min (µs)", "max (µs)"
    );
    println!(
        "|{:-^12}|{:-^14}|{:-^14}|{:-^14}|{:-^14}|{:-^14}|",
        "", "", "", "", "", ""
    );
    for (heads, c) in results {
        println!(
            "| {:>10} | {:>12.2} | {:>12.2} | {:>12.2} | {:>12.2} | {:>12.2} |",
            heads,
            c.mean.as_micros(),
            c.median.as_micros(),
            c.variance.as_nanos(),
            c.min.as_micros(),
            c.max.as_micros(),
        );
    }
    println!("| {:-<1$} |", "", 85);
    println!("");
}
