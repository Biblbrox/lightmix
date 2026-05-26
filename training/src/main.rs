#![recursion_limit = "2048"]

use std::{env::current_dir, fs::File, path::PathBuf};

use burn::{
    backend::Autodiff, grad_clipping::GradientClippingConfig, optim::AdamWConfig,
    tensor::backend::Backend,
};
use burn_cuda::Cuda;
use lightmix::{
    augmentations::{Pipeline, builder::AugmentationBuilder},
    config::{OptimizerConfig, ParsedConfig},
    data::dataset::{DatasetType, LazyFiletype},
    models::{
        efficientvit::EfficientViTConfig, fast_vit::FastViTConfig, fast_vit3d::FastViT3DConfig,
        vit::ViTConfig,
    },
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

fn match_dataset<B: Backend>(dataset_name: &str) -> DatasetType {
    dataset_name
        .parse::<DatasetType>()
        .expect("Unknown dataset")
}

fn run_experiment<B: Backend>(config: ParsedConfig, device: B::Device) {
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

    let ds_type = match_dataset::<B>(&dataset_name);
    let dataset = ds_type.make_dataset();
    let batcher_train = ds_type.make_batcher::<Autodiff<B>>();
    let batcher_val = ds_type.make_batcher::<B>();

    let (pipeline_train, pipeline_val): (Pipeline<Autodiff<B>>, Pipeline<B>) =
        AugmentationBuilder::new().build(
            &shared.augmentations,
            dataset_cfg.mean.clone(),
            dataset_cfg.std.clone(),
            &device,
        );

    match model_name.as_str() {
        name if name.starts_with("fast_vit_cloud") => {
            let model_cfg: FastViT3DConfig = model_table.try_into().unwrap();
            let artifact_dir = format!("./experiments/{}-{}", model_cfg.model_name(), dataset_name);
            train(
                &artifact_dir,
                LazyFiletype::Arrow,
                dataset_path.into(),
                shared,
                dataset_cfg,
                device,
                model_cfg,
                dataset,
                pipeline_train,
                pipeline_val,
                optimizer,
                batcher_train.clone(),
                batcher_val.clone(),
            );
        }
        name if name.starts_with("fast_vit") => {
            let model_cfg: FastViTConfig = model_table.try_into().unwrap();
            let artifact_dir = format!("./experiments/{}-{}", model_cfg.model_name(), dataset_name);
            train(
                &artifact_dir,
                LazyFiletype::Arrow,
                dataset_path.into(),
                shared,
                dataset_cfg,
                device,
                model_cfg,
                dataset,
                pipeline_train,
                pipeline_val,
                optimizer,
                batcher_train.clone(),
                batcher_val.clone(),
            );
        }
        name if name.starts_with("vit") => {
            let model_cfg: ViTConfig = model_table.try_into().unwrap();
            let artifact_dir = format!("./experiments/{}-{}", model_cfg.model_name(), dataset_name);
            train(
                &artifact_dir,
                LazyFiletype::Arrow,
                dataset_path.into(),
                shared,
                dataset_cfg,
                device,
                model_cfg,
                dataset,
                pipeline_train,
                pipeline_val,
                optimizer,
                batcher_train.clone(),
                batcher_val.clone(),
            );
        }
        name if name.starts_with("efficientvit") => {
            let model_cfg: EfficientViTConfig = model_table.try_into().unwrap();
            let artifact_dir = format!("./experiments/{}-{}", model_cfg.model_name(), dataset_name);
            train(
                &artifact_dir,
                LazyFiletype::Arrow,
                dataset_path.into(),
                shared,
                dataset_cfg,
                device,
                model_cfg,
                dataset,
                pipeline_train,
                pipeline_val,
                optimizer,
                batcher_train.clone(),
                batcher_val.clone(),
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
