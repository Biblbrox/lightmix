use std::{path::PathBuf, sync::Arc};

use burn::{
    backend::autodiff::checkpoint::strategy::CheckpointStrategy,
    lr_scheduler::{
        LrScheduler,
        cosine::{CosineAnnealingLrScheduler, CosineAnnealingLrSchedulerConfig},
        linear::{LinearLrScheduler, LinearLrSchedulerConfig},
    },
    module::Module,
    nn::{LinearRecord, loss::CrossEntropyLossConfig},
    optim::{AdamW, AdamWConfig, Optimizer, adaptor::OptimizerAdaptor},
    prelude::Backend,
    record::{
        CompactRecorder, DefaultRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Record,
        Recorder,
    },
    tensor::{Int, Tensor, backend::AutodiffBackend},
    train::{
        ClassificationOutput, EarlyStoppingStrategy, InferenceStep, Learner,
        LearnerOptimizerRecord, MetricEarlyStoppingStrategy, StoppingCondition, SupervisedTraining,
        TrainOutput, TrainStep, TrainingStrategy,
        checkpoint::CheckpointingStrategy,
        metric::{
            AccuracyMetric, Adaptor, CpuMemory, CpuTemperature, CudaMetric, LearningRateMetric,
            LossMetric, Metric, TopKAccuracyMetric,
        },
    },
};

use crate::{
    augmentations::{
        Augmentation, Pipeline,
        colors::{ColorJitter, RandomGrayscale},
        normalize::Normalize,
        rotation::{Orientation, RandomAffine, RandomFlip},
    },
    config::Config,
    data::{
        batch::{
            Batch, cifar10::Cifar10Batcher, cifar100::Cifar100Batcher,
            fashionmnist::FashionMnistBatcher, food101::Food101Batcher,
            imagenet1k::ImageNet1kBatcher, mnist::MnistBatcher, tinyimagenet::TinyImageNetBatcher,
        },
        builder::StreamingDataLoaderBuilder,
        dataset::{
            LazyDataset, LazyFiletype, cifar10::Cifar10Dataset, cifar100::Cifar100Dataset,
            fashionmnist::FashionMnistDataset, food101::Food101Dataset,
            imagenet1k::ImageNet1kDataset, mnist::MnistDataset, tinyimagenet::TinyImageNetDataset,
        },
        mapper::{
            cifar10::Cifar10Mapper, cifar100::Cifar100Mapper, fashionmnist::FashionMnistMapper,
            food101::Food101Mapper, imagenet1k::ImageNet1kMapper, mnist::MnistMapper,
            tinyimagenet::TinyImageNetMapper,
        },
        strategy::buffered::BufferedBatchStrategy,
    },
    models::spectre_vit::{SpectreViT as Model, SpectreViTConfig},
};

//type Dataset = FashionMnistDataset;
//type Batcher = FashionMnistBatcher;
//type Mapper = FashionMnistMapper;

type Dataset = TinyImageNetDataset;
type Batcher = TinyImageNetBatcher;
type Mapper = TinyImageNetMapper;

//type Dataset = Food101Dataset;
//type Batcher = Food101Batcher;
//type Mapper = Food101Mapper;

//type Dataset = Cifar100Dataset;
//type Batcher = Cifar100Batcher;
//type Mapper = Cifar100Mapper;

//type Dataset = Cifar10Dataset;
//type Batcher = Cifar10Batcher;
//type Mapper = Cifar10Mapper;

//type Dataset = ImageNet1kDataset;
//type Batcher = ImageNet1kBatcher;
//type Mapper = ImageNet1kMapper;

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = Batch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Batch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = Batch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Batch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &String,
    data_dir: &str,
    config: Config,
    device: B::Device,
    model: SpectreViTConfig,
    optimizer: AdamWConfig,
) {
    // Remove existing artifacts before to get an accurate learner summary
    if !config.continue_training {
        std::fs::remove_dir_all(artifact_dir).ok();
        std::fs::create_dir_all(artifact_dir).ok();
    }

    config.save(PathBuf::from(format!("{artifact_dir}/config.json")).as_path());

    B::seed(&device, config.random_seed as u64);

    let ds = Dataset::new(data_dir, LazyFiletype::Arrow);
    let batcher = Batcher::new();
    let strategy = BufferedBatchStrategy::new(
        config.batch_size as usize,
        config.batch_size as usize * 100,
        config.num_workers as usize,
    ); //.with_mapper(Mapper::decoder());

    let normalize = Box::new(Normalize::<B>::new(
        config.std.clone(),
        config.mean.clone(),
        &device,
    ));
    let normalize_val = Box::new(Normalize::<B::InnerBackend>::new(
        config.std,
        config.mean,
        &device,
    ));
    let random_flip_hor = Box::new(RandomFlip::<B>::new(0.5, Orientation::Horizontal));
    let random_flip_ver = Box::new(RandomFlip::<B>::new(0.5, Orientation::Vertical));
    let random_affine = Box::new(RandomAffine::<B>::new(0.5, 30.0));
    let color_jitter = Box::new(ColorJitter::<B>::new(0.4, 0.4, 0.1, &device));
    let random_gray = Box::new(RandomGrayscale::<B>::new(0.5, &device));

    let transforms_train: Vec<Box<dyn Augmentation<B>>> = vec![
        random_flip_hor,
        color_jitter,
        //random_gray,
        random_affine,
        random_flip_ver,
        normalize,
    ]; //, color_jitter, random_flip_hor, random_flip_ver];
    let pipeline_train = Pipeline::new(transforms_train);

    let transforms_val: Vec<Box<dyn Augmentation<B::InnerBackend>>> = vec![normalize_val];
    let pipeline_val = Pipeline::<B::InnerBackend>::new(transforms_val);

    let dataloader_train = StreamingDataLoaderBuilder::<B>::new(batcher.clone())
        .with_strategy(strategy.clone().with_shuffle(config.random_seed as u64))
        .with_transforms(Arc::new(pipeline_train))
        .with_device(device.clone())
        .build(ds.train());
    let dataloader_val = StreamingDataLoaderBuilder::<B::InnerBackend>::new(batcher.clone())
        .with_strategy(strategy)
        .with_transforms(Arc::new(pipeline_val))
        .with_device(device.clone())
        .build(ds.validation());

    let accuracy_metric = AccuracyMetric::new();
    let top5accuracy = TopKAccuracyMetric::new(5);
    let recorder = DefaultRecorder::new();
    let learner = SupervisedTraining::new(artifact_dir, dataloader_train.clone(), dataloader_val)
        .metrics((
            accuracy_metric.clone(),
            LossMetric::new(),
            top5accuracy,
            CpuTemperature::new(),
            LearningRateMetric::new(),
            CpuMemory::new(),
        ))
        .with_file_checkpointer(recorder.clone())
        .num_epochs(config.epochs as usize)
        .early_stopping(MetricEarlyStoppingStrategy::new(
            &accuracy_metric,
            burn::train::metric::store::Aggregate::Mean,
            burn::train::metric::store::Direction::Highest,
            burn::train::metric::store::Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 10 },
        ))
        .summary();

    let mut model = model.init::<B>(&device);
    let mut optimizer = optimizer.init();
    let mut scheduler = CosineAnnealingLrSchedulerConfig::new(
        config.learning_rate,
        config.epochs as usize * (dataloader_train.num_items() / config.batch_size as usize),
    )
    .init()
    .unwrap();

    if config.continue_training {
        let epoch = config.resume_epoch;
        let model_path = format!("{artifact_dir}/model-{epoch}.mpk");
        let optimizer_path = format!("{artifact_dir}/optim-{epoch}.mpk");
        let scheduler_path = format!("{artifact_dir}/scheduler-{epoch}.mpk");
        model = model.load_file(model_path, &recorder, &device).unwrap();
        optimizer = optimizer.load_record(recorder.load(optimizer_path.into(), &device).unwrap());
        scheduler = scheduler.load_record::<B>(
            <NamedMpkFileRecorder<FullPrecisionSettings> as Recorder<B>>::load::<usize>(
                &recorder,
                scheduler_path.into(),
                &device,
            )
            .unwrap(),
        );
    }

    let result = learner.launch(Learner::new(model, optimizer, scheduler));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &recorder)
        .expect("Trained model should be saved successfully");
}
