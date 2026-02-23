//mod inference;
mod model;
mod training;

mod config;
mod dataloader;
mod dataset;
mod kernels;
mod spectre_vit;
mod vit;
use std::env::current_dir;

use crate::{
    //model::ModelConfig as ModelConfig,
    config::Config,
    spectre_vit::SpectreViTConfig as ModelConfig,
    training::TrainingConfig,
};
use burn::{
    backend::{Autodiff, Cuda},
    optim::AdamWConfig,
};

fn main() {
    type MyBackend = Cuda<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let cwd = current_dir().unwrap();
    let path = cwd.join("experiments.toml");
    let localpath = cwd.join("experiments.local.toml");
    let config = Config::parse(&path, "mnist", "model", Some(&localpath));

    let device = burn::backend::cuda::CudaDevice::default();

    // let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(
            ModelConfig::new(
                config.embed_dim as usize,
                config.num_heads as usize,
                config.num_encoders as usize,
                config.num_classes as usize,
                config.patch_size as usize,
                config.img_size as usize,
                config.hidden_dim as usize,
                config.dropout,
            ),
            AdamWConfig::new(),
        )
        .with_batch_size(config.batch_size as usize)
        .with_num_epochs(config.epochs as usize),
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
