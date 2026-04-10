#![recursion_limit = "2048"]

mod augmentations;
mod config;
mod data;
mod mixing;
mod models;
mod norm;
mod tokenization;
mod training;
mod utils;

use std::{collections::HashMap, env::current_dir, fs::File, path::PathBuf};

use crate::{config::Config, models::spectre_vit::SpectreViTConfig as ModelConfig};
use burn::{
    backend::{Autodiff, Cuda, NdArray},
    optim::AdamWConfig,
};
use simplelog::{LevelFilter, SharedLogger, WriteLogger};
use tikv_jemallocator::Jemalloc;

fn init_logger() {
    WriteLogger::init(
        LevelFilter::Info,
        simplelog::Config::default(),
        File::create("training.log").unwrap(),
    )
    .unwrap();
}

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    type MyBackend = Cuda<f32, i32>;
    let device = burn::backend::cuda::CudaDevice::default();
    init_logger();

    //type MyBackend = Vulkan<f32, i32>;
    //let device = burn::backend::wgpu::WgpuDevice::default();

    type MyAutodiffBackend = Autodiff<MyBackend>;

    let cwd = current_dir().unwrap();
    let path = cwd.join("training/experiments.toml");
    let localpath = cwd.join("training/experiments.local.toml");
    if !path.exists() {
        eprintln!("Config path {} doesn't exist", path.to_str().unwrap());
    }
    if !localpath.exists() {
        eprintln!(
            "Local config path {} doesn't exist",
            localpath.to_str().unwrap()
        );
    }
    let dataset = "fashionmnist";
    let model_name = "spectre_vit";
    let config = Config::parse(&path, dataset, model_name, Some(&localpath));
    println!("Config loaded from path {}", path.to_str().unwrap());
    let dataset_path_buf = PathBuf::from(config.cache_dir.as_str()).join(dataset);
    if !dataset_path_buf.exists() {
        eprintln!(
            "Dataset path {} doesn't exist",
            dataset_path_buf.to_str().unwrap()
        );
    }
    let dataset_path = dataset_path_buf.as_path().to_str().unwrap();
    println!("Loading dataset from path {}", dataset_path);

    // let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = format!(
        "./assets/{}-{}-head{:?}-hid{:?}-emb{:?}-enc{:?}-weightedpermut",
        model_name,
        dataset,
        config.num_heads,
        config.hidden_dim,
        config.embed_dim,
        config.num_encoders
    );
    crate::training::train::<MyAutodiffBackend>(
        &artifact_dir,
        dataset_path,
        config.clone(),
        device.clone(),
        ModelConfig::new(
            config.in_channels as usize,
            config.embed_dim as usize,
            config.num_heads as usize,
            config.num_encoders as usize,
            config.num_classes as usize,
            config.patch_size as usize,
            config.img_size as usize,
            config.hidden_dim as usize,
            config.dropout,
            config.sinkhorn_temp as f32,
        ),
        AdamWConfig::new()
            .with_weight_decay(config.adam_weight_decay as f32)
            .with_beta_1(config.adam_betas[0] as f32)
            .with_beta_2(config.adam_betas[1] as f32),
    );

    //crate::inference::infer::<MyBackend>(
    //    artifact_dir,
    //    device,
    //    burn::data::dataset::vision::MnistDataset::test()
    //        .get(43)
    //        .unwrap(),
    //);

    // Print vit architecture
    //let model = ViTConfig::new(
    //    EMBED_DIM,
    //    NUM_HEADS,
    //    NUM_ENCODERS,
    //    NUM_CLASSES,
    //    PATCH_SIZE,
    //    IMG_SIZE,
    //);
    //println!("{model}");
}
