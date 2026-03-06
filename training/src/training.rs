use burn::{
    config::Config,
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::AdamWConfig,
    prelude::Backend,
    record::CompactRecorder,
    tensor::{Int, Tensor, backend::AutodiffBackend},
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

use crate::{
    data::{
        batch::{Batch, mnist::MnistBatcher},
        builder::StreamingDataLoaderBuilder,
        dataset::{LazyDataset, LazyFiletype, mnist::MnistDataset},
        mapper::mnist::MnistMapper,
        strategy::buffered::BufferedBatchStrategy,
    },
    spectre_vit::{SpectreViT as Model, SpectreViTConfig as ModelConfig},
};

type Batch<B> = ImageNet1kBatch<B>;
type Dataset = ImageNet1kDataset;
type Batcher = ImageNet1kBatcher;
type Mapper = ImageNet1kMapper;

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
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
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

    // let cache_dir: PlRefPath = "/home/biblbrox/.cache/huggingface/hub".into();
    let mnist_path = "hf://datasets/ylecun/mnist";
    let mnist_ds = MnistDataset::new(mnist_path, LazyFiletype::Parquet);
    let mnist_batcher = MnistBatcher::new();
    let strategy =
        BufferedBatchStrategy::new(config.batch_size, 10).with_mapper(MnistMapper::decoder());

    let dataloader_train = StreamingDataLoaderBuilder::new(mnist_batcher.clone())
        .with_strategy(strategy.clone().with_shuffle(config.seed))
        .build(mnist_ds.train());
    let dataloader_test = StreamingDataLoaderBuilder::new(mnist_batcher.clone())
        .with_strategy(strategy)
        .build(mnist_ds.validation());

    let dataloader_train = StreamingDataLoaderBuilder::new(batcher.clone())
        .with_batch_size(config.batch_size)
        .with_shuffle(42)
        .with_device(device.clone())
        //.with_batch_mapper(Mapper::decoder())
        .build(dataset.train());

    let dataloader_test = StreamingDataLoaderBuilder::new(batcher.clone())
        .with_batch_size(config.val_batch_size)
        //.with_batch_mapper(Mapper::decoder())
        .with_device(device.clone())
        .build(dataset.test());

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init::<B>(&device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
