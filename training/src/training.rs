use std::sync::Arc;

use burn::{
    backend::autodiff::checkpoint::strategy::CheckpointStrategy,
    config::Config,
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::AdamWConfig,
    prelude::Backend,
    record::{CompactRecorder, DefaultRecorder},
    tensor::{Int, Tensor, backend::AutodiffBackend},
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        checkpoint::CheckpointingStrategy,
        metric::{
            AccuracyMetric, CpuMemory, CpuTemperature, CudaMetric, LossMetric, TopKAccuracyMetric,
        },
    },
};

use crate::{
    augmentations::{
        Augmentation, Pipeline,
        colors::ColorJitter,
        normalize::Normalize,
        rotation::{Orientation, RandomAffine, RandomFlip},
    },
    data::{
        batch::{
            Batch, cifar100::Cifar100Batcher, imagenet1k::ImageNet1kBatcher, mnist::MnistBatcher,
        },
        builder::StreamingDataLoaderBuilder,
        dataset::{
            LazyDataset, LazyFiletype, cifar100::Cifar100Dataset, imagenet1k::ImageNet1kDataset,
            mnist::MnistDataset,
        },
        mapper::{cifar100::Cifar100Mapper, imagenet1k::ImageNet1kMapper, mnist::MnistMapper},
        strategy::buffered::BufferedBatchStrategy,
    },
    spectre_vit::{SpectreViT as Model, SpectreViTConfig as ModelConfig},
};

type Dataset = Cifar100Dataset;
type Batcher = Cifar100Batcher;
type Mapper = Cifar100Mapper;

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

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 128)]
    pub val_batch_size: usize,
    #[config(default = 16)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &String,
    data_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let ds = Dataset::new(data_dir, LazyFiletype::Arrow);
    let batcher = Batcher::new();
    let strategy = BufferedBatchStrategy::new(config.batch_size, 100, config.num_workers); //.with_mapper(Mapper::decoder());
    // Imagenet1k normalize
    //let std = [0.229, 0.224, 0.225];
    //let mean = [0.485, 0.456, 0.406];

    let std = [0.2675, 0.2565, 0.2761];
    let mean = [0.5071, 0.4867, 0.4408];

    let normalize = Box::new(Normalize::<B, 3>::new(std, mean, &device));
    let normalize_val = Box::new(Normalize::<B::InnerBackend, 3>::new(std, mean, &device));
    let random_flip = Box::new(RandomFlip::<B>::new(0.5, Orientation::Horizontal));
    let color_jitter = Box::new(ColorJitter::<B, 3>::new(0.4, 0.4, 0.1, &device));

    let transforms_train: Vec<Box<dyn Augmentation<B>>> =
        vec![normalize, color_jitter, random_flip];
    let pipeline_train = Pipeline::new(transforms_train);

    let transforms_val: Vec<Box<dyn Augmentation<B::InnerBackend>>> = vec![normalize_val];
    let pipeline_val = Pipeline::<B::InnerBackend>::new(transforms_val);

    let dataloader_train = StreamingDataLoaderBuilder::<B>::new(batcher.clone())
        .with_strategy(strategy.clone().with_shuffle(config.seed))
        .with_transforms(Arc::new(pipeline_train))
        .with_device(device.clone())
        .build(ds.train());
    let dataloader_val = StreamingDataLoaderBuilder::<B::InnerBackend>::new(batcher.clone())
        .with_strategy(strategy)
        .with_transforms(Arc::new(pipeline_val))
        .with_device(device.clone())
        .build(ds.validation());

    let learner = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_val)
        .metrics((
            AccuracyMetric::new(),
            LossMetric::new(),
            TopKAccuracyMetric::new(5),
            CpuTemperature::new(),
            CpuMemory::new(),
        ))
        .with_file_checkpointer(DefaultRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init::<B>(&device);
    let result = learner.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
