#![recursion_limit = "2048"]

use std::{env::current_dir, fs::File, path::PathBuf};

use burn::{
    backend::Autodiff, grad_clipping::GradientClippingConfig, optim::AdamWConfig,
    tensor::backend::Backend,
};
use burn_cuda::Cuda;
use lightmix::{
    augmentations::{Pipeline, builder::AugmentationBuilder, normalize::Normalize},
    config::{OptimizerConfig, ParsedConfig},
    data::dataset::{LazyFiletype, cifar100::Cifar100Dataset},
    models::{efficientvit::EfficientViTConfig, fast_vit::FastViTConfig, vit::ViTConfig},
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

pub fn run_experiment<B: Backend>(config: ParsedConfig, device: B::Device) {
    let optimizer_cfg: OptimizerConfig = config.optimizer();
    let ParsedConfig {
        shared,
        dataset: dataset_cfg,
        model_table,
    } = config;

    let dataset_name = shared.active_dataset.clone();
    let model_name = shared.active_model.clone();

    let dataset_path = PathBuf::from(&shared.cache_dir).join(&dataset_name);
    if !dataset_path.exists() {
        eprintln!("Dataset path {} doesn't exist", dataset_path.display());
    }
    let dataset_path = dataset_path.to_str().unwrap();

    let optimizer = AdamWConfig::new()
        .with_weight_decay(optimizer_cfg.adam_weight_decay as f32)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .with_beta_1(optimizer_cfg.adam_betas[0] as f32)
        .with_beta_2(optimizer_cfg.adam_betas[1] as f32);

    let normalize_train = Box::new(Normalize::<Autodiff<B>>::new(
        dataset_cfg.std.clone(),
        dataset_cfg.mean.clone(),
        &device,
    ));
    let normalize_val = Box::new(Normalize::<B>::new(
        dataset_cfg.std.clone(),
        dataset_cfg.mean.clone(),
        &device,
    ));
    let mut pipeline_train = AugmentationBuilder::<Autodiff<B>>::new(device.clone()).build(
        &shared.augmentations,
        dataset_cfg.mean.clone(),
        dataset_cfg.std.clone(),
    );
    pipeline_train = pipeline_train.prepend(vec![normalize_train]);
    let pipeline_val = Pipeline::<B>::new(vec![normalize_val]);

    let dataset = Cifar100Dataset::new(dataset_path, LazyFiletype::Arrow);

    match model_name.as_str() {
        name if name.starts_with("fast_vit") => {
            let model_cfg: FastViTConfig = model_table.try_into().unwrap();
            let artifact_dir = format!(
                "./assets/{}-{}-head{}-hid{}-emb{}-enc{}-temp{}-learnedmixer",
                model_name,
                dataset_name,
                model_cfg.num_heads,
                model_cfg.hidden_dim,
                model_cfg.embed_dim,
                model_cfg.num_encoders,
                model_cfg.sinkhorn_temp,
            );
            train(
                &artifact_dir,
                shared,
                dataset_cfg,
                device,
                model_cfg,
                dataset,
                pipeline_train,
                pipeline_val,
                optimizer,
            );
        }
        name if name.starts_with("vit") => {
            let model_cfg: ViTConfig = model_table.try_into().unwrap();
            let artifact_dir = format!(
                "./assets/{}-{}-head{}-hid{}-emb{}-enc{}",
                model_name,
                dataset_name,
                model_cfg.num_heads,
                model_cfg.hidden_dim,
                model_cfg.embed_dim,
                model_cfg.num_encoders,
            );
            train(
                &artifact_dir,
                shared,
                dataset_cfg,
                device,
                model_cfg,
                dataset,
                pipeline_train,
                pipeline_val,
                optimizer,
            );
        }
        name if name.starts_with("efficientvit") => {
            let model_cfg: EfficientViTConfig = model_table.try_into().unwrap();
            let artifact_dir = format!(
                "./assets/{}-{}-stem{}-ch{:?}-dep{:?}",
                model_name,
                dataset_name,
                model_cfg.stem_channels,
                model_cfg.stage_channels,
                model_cfg.stage_depths,
            );
            train(
                &artifact_dir,
                shared,
                dataset_cfg,
                device,
                model_cfg,
                dataset,
                pipeline_train,
                pipeline_val,
                optimizer,
            );
        }
        _ => panic!("Unknown model: {}", model_name),
    }
}

fn main() {
    type MyBackend = Cuda<f32, i32>;
    let device = burn::backend::cuda::CudaDevice::default();

    init_logger();

    let cwd = current_dir().unwrap();
    let path = cwd.join("training/experiments.toml");
    let localpath = cwd.join("training/experiments.local.toml");

    if !path.exists() {
        eprintln!("Config path {} doesn't exist", path.display());
    }
    if !localpath.exists() {
        eprintln!("Local config path {} doesn't exist", localpath.display());
    }

    let config = ParsedConfig::parse(&path, Some(&localpath));
    println!("Config loaded from {}", path.display());

    run_experiment::<MyBackend>(config, device);
}
