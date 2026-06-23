#![recursion_limit = "2048"]

use std::{fs::File, io::Write, panic, path::PathBuf};

use lightmix::{config::ParsedConfig, training::run_experiment};

#[cfg(feature = "jemalloc")]
use tikv_jemallocator::Jemalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    type MyBackend = burn::backend::cuda::Cuda<f32, i32>;
    let device = burn::backend::cuda::CudaDevice::default();

    let config_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../configs");
    let localpath = config_dir.join("experiments.local.toml");

    let config = ParsedConfig::load(&config_dir, Some(&localpath));
    println!("Config loaded from {}", config_dir.display());

    panic::set_hook(Box::new(move |info| {
        let backtrace = std::backtrace::Backtrace::capture();

        let msg = format!("=== PANIC ===\n{info}\n\n=== BACKTRACE ===\n{backtrace}\n",);

        if let Ok(mut f) = File::create("panic.log") {
            let _ = f.write_all(msg.as_bytes());
        }

        eprintln!("{msg}");
    }));

    run_experiment::<MyBackend>(config, device);
}
