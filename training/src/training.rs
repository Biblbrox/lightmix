use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

use burn::{
    backend::Autodiff,
    data::dataloader::Progress,
    lr_scheduler::{LrScheduler, cosine::CosineAnnealingLrSchedulerConfig},
    module::{AutodiffModule, Module},
    optim::{AdamWConfig, Optimizer},
    record::{DefaultRecorder, Recorder},
    tensor::backend::Backend,
    train::{
        InferenceStep, Interrupter, LearnerSummary, TrainStep,
        logger::{FileMetricLogger, MetricLogger},
        metric::{
            AccuracyInput, AccuracyMetric, LossInput, LossMetric, MetricMetadata,
            TopKAccuracyInput, TopKAccuracyMetric,
            store::{EpochSummary, Split},
        },
        renderer::tui::TuiMetricsRendererWrapper,
    },
};
use polars::prelude::PlRefPath;
use serde::Serialize;

use crate::{
    augmentations::{Pipeline, builder::AugmentationBuilder},
    data::dataset::{DatasetType, LazyDataset, LazyFiletype},
    metrics::MetricsHandler,
    models::ModelConfig,
};
use crate::{config::DatasetConfig, data::dataloader::strategy::buffered::BufferedBatchStrategy};
use crate::{config::SharedConfig, data::builder::StreamingDataLoaderBuilder};

pub trait Saveable: Serialize {
    fn save(&self, path: &Path) {
        let mut file = File::create(path).unwrap();
        let content = toml::to_string_pretty(self).unwrap();
        file.write_all(content.as_bytes()).unwrap();
    }
}

impl Saveable for SharedConfig {}
impl Saveable for DatasetConfig {}

pub fn train<B: Backend>(
    artifact_dir: &str,
    dataset_type: LazyFiletype,
    dataset_path: PlRefPath,
    shared: SharedConfig,
    dataset_cfg: DatasetConfig,
    device: B::Device,
    model: impl ModelConfig<B>,
    dataset: DatasetType,
    optimizer: AdamWConfig,
) {
    // Remove existing artifacts before to get an accurate learner summary
    if !shared.continue_training {
        std::fs::remove_dir_all(artifact_dir).ok();
        std::fs::create_dir_all(artifact_dir).ok();
    }

    let batcher_train = dataset.make_batcher::<Autodiff<B>>();
    let batcher_val = dataset.make_batcher::<B>();
    let dataset = dataset.make_dataset();

    let (pipeline_train, pipeline_val): (Pipeline<Autodiff<B>>, Pipeline<B>) =
        AugmentationBuilder::new().build(
            &shared.augmentations,
            dataset_cfg.mean.clone(),
            dataset_cfg.std.clone(),
            &device,
        );

    shared.save(PathBuf::from(format!("{artifact_dir}/shared_config.json")).as_path());
    dataset_cfg.save(PathBuf::from(format!("{artifact_dir}/dataset_config.json")).as_path());

    B::seed(&device, shared.random_seed as u64);

    let strategy = BufferedBatchStrategy::new(
        dataset_cfg.batch_size,
        dataset_cfg.batch_size,
        shared.num_workers as usize,
    );

    let dataloader_train = StreamingDataLoaderBuilder::<Autodiff<B>>::new(batcher_train)
        .with_strategy(strategy.clone().with_shuffle(shared.random_seed as u64))
        .with_transforms(Arc::new(pipeline_train))
        .with_device(device.clone())
        .build(dataset.train(dataset_path.clone(), dataset_type.clone()));
    let dataloader_val = StreamingDataLoaderBuilder::<B>::new(batcher_val)
        .with_strategy(strategy)
        .with_transforms(Arc::new(pipeline_val))
        .with_device(device.clone())
        .build(dataset.validation(dataset_path, dataset_type));

    let recorder = DefaultRecorder::new();

    let num_iterations = dataloader_train.num_items() / dataset_cfg.batch_size;
    let mut model = model.init_training(
        &device,
        dataset_cfg.in_channels,
        dataset_cfg.img_size,
        dataset_cfg.num_classes,
    );
    let mut optimizer = optimizer.init();
    let mut scheduler = CosineAnnealingLrSchedulerConfig::new(
        shared.learning_rate,
        dataset_cfg.epochs * num_iterations,
    )
    .init()
    .unwrap();

    let mut train_metrics = MetricsHandler::<Autodiff<B>>::new()
        .add(LossMetric::new(), |o| LossInput::new(o.loss()))
        .add(AccuracyMetric::new(), |o| {
            AccuracyInput::new(o.output(), o.targets())
        })
        .add(TopKAccuracyMetric::new(5), |o| {
            TopKAccuracyInput::new(o.output(), o.targets())
        });
    let mut valid_metrics = MetricsHandler::<B>::new()
        .add(LossMetric::new(), |o| LossInput::new(o.loss()))
        .add(AccuracyMetric::new(), |o| {
            AccuracyInput::new(o.output(), o.targets())
        })
        .add(TopKAccuracyMetric::new(5), |o| {
            TopKAccuracyInput::new(o.output(), o.targets())
        });

    let mut stop_flag = false;
    let mut logger = FileMetricLogger::new(artifact_dir);
    for definition in train_metrics.definitions() {
        logger.log_metric_definition(definition.clone());
    }

    #[cfg(feature = "viz-rerun")]
    let mut rerun_rec: Option<rerun::RecordingStream> = None;

    let interrupter = Interrupter::new();
    let mut renderer = TuiMetricsRendererWrapper::new(interrupter.clone(), None);
    valid_metrics.register(&mut renderer);
    train_metrics.register(&mut renderer);

    for epoch in 1..=dataset_cfg.epochs {
        let global_progress = Progress {
            items_processed: epoch,
            items_total: dataset_cfg.epochs,
        };

        #[cfg(feature = "viz-rerun")]
        if rerun_rec.is_none() {
            rerun_rec = Some(
                rerun::RecordingStreamBuilder::new("lightmix")
                    .save(format!("{artifact_dir}/validation.rrd"))
                    .expect("Failed to create rerun recording stream"),
            );
        }

        let mut lr = 0.0;
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            if interrupter.should_stop() {
                stop_flag = true;
                break;
            }

            let progress = Progress {
                items_processed: iteration + 1,
                items_total: num_iterations,
            };

            let step_output = model.step(batch);
            let output = step_output.item;
            let grads = step_output.grads;

            lr = scheduler.step();
            model = optimizer.step(lr, model, grads);

            // Update metrics
            let metrics_metadata = MetricMetadata {
                progress: progress.clone(),
                global_progress: global_progress.clone(),
                iteration: Some(iteration),
                lr: Some(lr),
            };

            train_metrics.update_train(
                &output,
                &metrics_metadata,
                &mut renderer,
                &mut logger,
                epoch,
            );
            train_metrics.render_train(
                &mut renderer,
                &progress,
                &global_progress,
                iteration,
                epoch,
            );
        }

        let model_valid = model.valid();
        let num_val_iterations = dataloader_val.num_items() / dataset_cfg.batch_size;

        for (iteration, batch) in dataloader_val.iter().enumerate() {
            if interrupter.should_stop() {
                stop_flag = true;
                break;
            }

            let progress = Progress {
                items_processed: iteration + 1,
                items_total: num_val_iterations,
            };

            #[cfg(feature = "viz-rerun")]
            let batch_clone = batch.clone();

            let step_output = model_valid.step(batch);

            #[cfg(feature = "viz-rerun")]
            if let Some(ref rec) = rerun_rec {
                use crate::logging::log_3d_sample;

                log_3d_sample(&step_output, &batch_clone, epoch, iteration as i64, rec);
            }

            let metrics_metadata = MetricMetadata {
                progress: progress.clone(),
                global_progress: global_progress.clone(),
                iteration: Some(iteration),
                lr: Some(lr),
            };

            valid_metrics.update_valid(
                &step_output,
                &metrics_metadata,
                &mut renderer,
                &mut logger,
                epoch,
            );
            valid_metrics.render_valid(
                &mut renderer,
                &progress,
                &global_progress,
                iteration,
                epoch,
            );
        }

        model
            .clone()
            .save_file(format!("{artifact_dir}/model-epoch-{epoch}"), &recorder)
            .expect("Checkpoint save failed");

        <DefaultRecorder as Recorder<Autodiff<B>>>::record(
            &recorder,
            optimizer.to_record(),
            format!("{artifact_dir}/optim-epoch-{epoch}").into(),
        )
        .ok();

        <DefaultRecorder as Recorder<B>>::record(
            &recorder,
            scheduler.to_record::<B>(),
            format!("{artifact_dir}/scheduler-epoch-{epoch}").into(),
        )
        .ok();

        logger.log_epoch_summary(EpochSummary {
            epoch_number: epoch,
            split: Split::Train,
        });
        logger.log_epoch_summary(EpochSummary {
            epoch_number: epoch,
            split: Split::Valid,
        });

        if stop_flag {
            interrupter.stop(Some("Training finished"));
            break;
        }
    }

    #[cfg(feature = "viz-rerun")]
    if let Some(rec) = rerun_rec.take() {
        drop(rec);
    }

    // If we don't do that, renderer won't allow stdout to pass
    drop(renderer);

    let metric_names = train_metrics.metric_names();
    println!("{}", model);
    match LearnerSummary::new(artifact_dir, &metric_names) {
        Ok(summary) => eprintln!("{}", summary),
        Err(e) => eprintln!("Summary unavailable: {}", e),
    }
}
