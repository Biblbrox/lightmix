use std::{path::PathBuf, sync::Arc};

use burn::{
    backend::Autodiff,
    data::dataloader::Progress,
    lr_scheduler::{LrScheduler, cosine::CosineAnnealingLrSchedulerConfig},
    module::{AutodiffModule, Module},
    optim::{AdamWConfig, Optimizer},
    record::{DefaultRecorder, Recorder},
    tensor::backend::{AutodiffBackend, Backend},
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

use crate::data::builder::StreamingDataLoaderBuilder;
use crate::{
    augmentations::builder::AugmentationBuilder, data::batch::tinyimagenet::TinyImageNetBatcher,
};
use crate::{
    augmentations::{Pipeline, normalize::Normalize},
    config::Config,
    data::{
        batch::cifar100::Cifar100Batcher, dataset::LazyDataset,
        strategy::buffered::BufferedBatchStrategy,
    },
    metrics::MetricsHandler,
    models::ModelConfig,
};

type Batcher = Cifar100Batcher;

pub fn train<B: Backend>(
    artifact_dir: &String,
    config: Config,
    device: B::Device,
    model: impl ModelConfig<B>,
    dataset: impl LazyDataset,
    optimizer: AdamWConfig,
) {
    // Remove existing artifacts before to get an accurate learner summary
    if !config.continue_training {
        std::fs::remove_dir_all(artifact_dir).ok();
        std::fs::create_dir_all(artifact_dir).ok();
    }

    config.save(PathBuf::from(format!("{artifact_dir}/config.json")).as_path());

    B::seed(&device, config.random_seed as u64);

    let batcher = Batcher::new();
    let strategy = BufferedBatchStrategy::new(
        config.batch_size as usize,
        config.batch_size as usize,
        config.num_workers as usize,
    );

    let normalize = Box::new(Normalize::<Autodiff<B>>::new(
        config.clone().std,
        config.clone().mean,
        &device.clone(),
    ));
    let normalize_val = Box::new(Normalize::<B>::new(
        config.clone().std,
        config.clone().mean,
        &device.clone(),
    ));
    let mut pipeline_train = AugmentationBuilder::new(device.clone()).build_from_config(&config);
    // Dirty hack to avoid adding normalization into augmentation pipeline. It will be added later
    pipeline_train = pipeline_train.prepend(vec![normalize]);
    let pipeline_val = Pipeline::<B>::new(vec![normalize_val]);

    let dataloader_train = StreamingDataLoaderBuilder::<Autodiff<B>>::new(batcher.clone())
        .with_strategy(strategy.clone().with_shuffle(config.random_seed as u64))
        .with_transforms(Arc::new(pipeline_train))
        .with_device(device.clone())
        .build(dataset.train());
    let dataloader_val = StreamingDataLoaderBuilder::<B>::new(batcher.clone())
        .with_strategy(strategy)
        .with_transforms(Arc::new(pipeline_val))
        .with_device(device.clone())
        .build(dataset.validation());

    //let dataloader_train = InMemoryDataLoaderBuilder::<Autodiff<B>>::new(batcher.clone())
    //    .with_transforms(Arc::new(pipeline_train))
    //    .with_device(device.clone())
    //    .with_num_workers(config.num_workers as usize)
    //    .with_batch_size(config.batch_size as usize)
    //    .build(dataset.train());
    //let dataloader_val = InMemoryDataLoaderBuilder::<B>::new(batcher.clone())
    //    .with_transforms(Arc::new(pipeline_val))
    //    .with_num_workers(config.num_workers as usize)
    //    .with_batch_size(config.batch_size as usize)
    //    .with_device(device.clone())
    //    .build(dataset.validation());

    let recorder = DefaultRecorder::new();
    let num_items = dataloader_train.num_items();

    let mut model = model.init_training(&device);
    let mut optimizer = optimizer.init();
    let mut scheduler = CosineAnnealingLrSchedulerConfig::new(
        config.learning_rate,
        config.epochs as usize * (num_items / config.batch_size as usize),
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

    let num_iterations = dataloader_train.num_items() / config.batch_size as usize;
    let mut stop_flag = false;
    let mut logger = FileMetricLogger::new(artifact_dir);
    for definition in train_metrics.definitions() {
        logger.log_metric_definition(definition.clone());
    }

    let interrupter = Interrupter::new();
    let mut renderer = TuiMetricsRendererWrapper::new(interrupter.clone(), None);
    valid_metrics.register(&mut renderer);
    train_metrics.register(&mut renderer);

    for epoch in 1..=config.epochs {
        let global_progress = Progress {
            items_processed: epoch as usize,
            items_total: config.epochs as usize,
        };

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
                epoch as usize,
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
        let num_val_iterations = dataloader_val.num_items() / config.batch_size as usize;

        for (iteration, batch) in dataloader_val.iter().enumerate() {
            if interrupter.should_stop() {
                stop_flag = true;
                break;
            }

            let progress = Progress {
                items_processed: iteration + 1,
                items_total: num_val_iterations,
            };

            let step_output = model_valid.step(batch);
            let output = step_output;

            let metrics_metadata = MetricMetadata {
                progress: progress.clone(),
                global_progress: global_progress.clone(),
                iteration: Some(iteration),
                lr: Some(lr),
            };

            valid_metrics.update_valid(
                &output,
                &metrics_metadata,
                &mut renderer,
                &mut logger,
                epoch as usize,
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
            epoch_number: epoch as usize,
            split: Split::Train,
        });
        logger.log_epoch_summary(EpochSummary {
            epoch_number: epoch as usize,
            split: Split::Valid,
        });

        if stop_flag {
            interrupter.stop(Some("Training finished"));
            break;
        }
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
