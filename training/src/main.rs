#![recursion_limit = "2048"]

mod model;
mod norm;
mod training;
mod utils;

mod augmentations;
mod config;
mod data;
mod spectre_vit;
mod vit;
use std::{env::current_dir, path::PathBuf};

use crate::{
    config::Config, spectre_vit::SpectreViTConfig as ModelConfig, training::TrainingConfig,
};
use burn::{
    backend::{Autodiff, Cuda, NdArray},
    optim::AdamWConfig,
};
use burn_wgpu::{Vulkan, Wgpu, WgpuDevice};
use tikv_jemallocator::Jemalloc;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;
fn main() {
    type MyBackend = Cuda<f32, i32>;
    let device = burn::backend::cuda::CudaDevice::default();
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
    let dataset = "cifar100";
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
        "./assets/{}-{}-head{:?}-hid{:?}-emb{:?}-enc{:?}",
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
        TrainingConfig::new(
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
            ),
            AdamWConfig::new()
                .with_weight_decay(config.adam_weight_decay as f32)
                .with_beta_1(config.adam_betas[0] as f32)
                .with_beta_2(config.adam_betas[1] as f32),
        )
        .with_batch_size(config.batch_size as usize)
        .with_val_batch_size(config.val_batch_size as usize)
        .with_num_epochs(config.epochs as usize)
        .with_num_workers(config.num_workers as usize)
        .with_learning_rate(config.learning_rate),
        device.clone(),
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
