mod data;
mod inference;
mod model;
mod training;

mod config;
mod dataloader;
mod dataset;
mod spectre_vit;
mod vit;
use crate::{
    //model::ModelConfig as ModelConfig,
    spectre_vit::SpectreViTConfig as ModelConfig,
    training::TrainingConfig,
};
use burn::{
    backend::{Autodiff, Cuda},
    optim::{AdamConfig, AdamWConfig},
    tensor::{backend::DeviceOps, f16},
};

fn main() {
    type MyBackend = Cuda<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    const NUM_CLASSES: usize = 10;
    const PATCH_SIZE: usize = 4;
    const IMG_SIZE: usize = 28;
    const NUM_HEADS: usize = 8;
    const NUM_ENCODERS: usize = 4;
    const EMBED_DIM: usize = PATCH_SIZE.pow(2) * 1 as usize;
    const HIDDEN_DIM: usize = 32;
    const DROPOUT: f64 = 0.1;

    // Training params
    const BATCH_SIZE: usize = 16;
    const EPOCHS: usize = 40;

    let device = burn::backend::cuda::CudaDevice::default();

    // let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(
            ModelConfig::new(
                EMBED_DIM,
                NUM_HEADS,
                NUM_ENCODERS,
                NUM_CLASSES,
                PATCH_SIZE,
                IMG_SIZE,
                HIDDEN_DIM,
                DROPOUT,
            ),
            AdamWConfig::new(),
        )
        .with_batch_size(BATCH_SIZE)
        .with_num_epochs(EPOCHS),
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
