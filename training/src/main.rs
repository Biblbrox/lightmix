#![recursion_limit = "2048"]

use std::{env::current_dir, fs::File, path::PathBuf};

use burn::{grad_clipping::GradientClippingConfig, optim::AdamWConfig};
use burn_cuda::Cuda;
use embed_former_train::{
    config::Config,
    data::dataset::{LazyFiletype, cifar100::Cifar100Dataset, tinyimagenet::TinyImageNetDataset},
    models::{fast_vit::FastViTConfig, vit::ViTConfig},
    training::train,
};
use simplelog::{LevelFilter, WriteLogger};
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
    let model_name = "spectre_vit_tiny";
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

    let artifact_dir = format!(
        "./assets/{}-{}-head{:?}-hid{:?}-emb{:?}-enc{:?}-temp-{}-learnedmixer",
        model_name,
        dataset,
        config.num_heads,
        config.hidden_dim,
        config.embed_dim,
        config.num_encoders,
        config.sinkhorn_temp,
    );

    //let e = config.embed_dim as usize;
    //let model_config = ModelConfig::new(
    //    config.in_channels as usize,
    //    config.num_classes as usize,
    //    config.patch_size as usize,
    //    config.img_size as usize,
    //    config.dropout,
    //    config.sinkhorn_temp as f32,
    //    4,
    //    vec![
    //        e,
    //        e,
    //        e,
    //        e * 2,
    //        e * 2,
    //        e * 2,
    //        e * 4,
    //        e * 4,
    //        e * 4,
    //        e * 4,
    //        e * 4,
    //        e * 4,
    //    ],
    //    vec![
    //        e * 4,
    //        e * 4,
    //        e * 4,
    //        e * 8,
    //        e * 8,
    //        e * 8,
    //        e * 16,
    //        e * 16,
    //        e * 16,
    //        e * 16,
    //        e * 16,
    //        e * 16,
    //    ],
    //    vec![4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8],
    //);

    train::<MyBackend>(
        &artifact_dir,
        config.clone(),
        device.clone(),
        // FastViTConfig::new(
        //     config.in_channels as usize,
        //     config.embed_dim as usize,
        //     config.num_heads as usize,
        //     config.num_encoders as usize,
        //     config.num_classes as usize,
        //     config.patch_size as usize,
        //     config.img_size as usize,
        //     config.hidden_dim as usize,
        //     config.dropout,
        //     config.sinkhorn_temp as f32,
        // ),
        ViTConfig::new(
            config.in_channels as usize,
            config.embed_dim as usize,
            config.hidden_dim as usize,
            config.num_heads as usize,
            config.num_encoders as usize,
            config.num_classes as usize,
            config.patch_size as usize,
            config.img_size as usize,
            config.dropout,
        ),
        Cifar100Dataset::new(dataset_path, LazyFiletype::Arrow),
        //model_config,
        AdamWConfig::new()
            .with_weight_decay(config.adam_weight_decay as f32)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
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
