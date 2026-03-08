pub mod cifar100;
pub mod imagenet1k;
pub mod mnist;

use polars::prelude::*;

pub enum LazyFiletype {
    Parquet,
    Arrow,
    Csv,
}

pub trait LazyDataset {
    fn train(&self) -> LazyFrame;
    fn validation(&self) -> LazyFrame;
    fn test(&self) -> LazyFrame;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    use crate::data::batch::cifar100::Cifar100Batcher;
    use crate::data::batch::imagenet1k::{ImageNet1kBatch, ImageNet1kBatcher};
    use crate::data::batch::mnist::MnistBatcher;
    use crate::data::builder::StreamingDataLoaderBuilder;
    use crate::data::dataset::LazyDataset;
    use crate::data::dataset::cifar100::Cifar100Dataset;
    use crate::data::dataset::imagenet1k::ImageNet1kDataset;
    use crate::data::dataset::mnist::MnistDataset;
    use crate::data::mapper::cifar100::Cifar100Mapper;
    use crate::data::mapper::imagenet1k::ImageNet1kMapper;
    use crate::data::mapper::mnist::MnistMapper;
    use crate::data::strategy::buffered::BufferedBatchStrategy;
    use burn::backend::autodiff::checkpoint::strategy;
    use burn::data::dataloader::DataLoader;
    use burn_cuda::{Cuda, CudaDevice};
    use indicatif::ProgressBar;
    use polars::prelude::{IpcScanOptions, PlRefPath, UnifiedScanArgs};

    //use tikv_jemallocator::Jemalloc;

    //#[global_allocator]
    //static GLOBAL: Jemalloc = Jemalloc;

    #[test]
    fn test_imagenet1k() {
        //let imagenet1k_path: PlRefPath =
        //    "/storage/experiments-ml/datasets/datasets--ILSVRC--imagenet-1k".into();

        // let imagenet1k_path: PlRefPath = "/storage2/Datasets/imagenet1k".into();
        let imagenet1k_path: PlRefPath = "/storage/experiments-ml/datasets/imagenet1k".into();

        let shuffle_seed = 42;
        let batch_size = 128;

        type B = Cuda;
        let device = CudaDevice::default();

        let ds = ImageNet1kDataset::new(imagenet1k_path, crate::data::dataset::LazyFiletype::Arrow);

        let batcher = ImageNet1kBatcher::new();
        let strategy = BufferedBatchStrategy::new(batch_size, 10); //.with_mapper(Mapper::decoder());
        let dl = StreamingDataLoaderBuilder::<B>::new(batcher.clone())
            .with_strategy(strategy.clone().with_shuffle(shuffle_seed))
            .with_device(device)
            .build(ds.test());

        let pbar = ProgressBar::new(dl.num_items() as u64);
        let start = Instant::now();
        //for (idx, _df) in imagenet1k_train_dl.iter().enumerate() {
        //    if idx >= 200 {
        //        break;
        //    }
        //    pbar.inc(batch_size as u64);
        //}
        for _df in dl.iter() {
            pbar.inc(batch_size as u64);
        }
        let duration = start.elapsed();
        pbar.finish_with_message("Done");
        println!("ImageNet1k train dataset preparing time: {:?}", duration);
    }

    #[test]
    fn test_cifar100() {
        let cifar100_path: PlRefPath = "/storage/experiments-ml/datasets/cifar100".into();

        let shuffle_seed = 42;
        let batch_size = 128;

        type B = Cuda;
        let device = CudaDevice::default();

        let ds = Cifar100Dataset::new(cifar100_path, crate::data::dataset::LazyFiletype::Arrow);
        let batcher = Cifar100Batcher::new();
        let strategy = BufferedBatchStrategy::new(batch_size, 10);

        let dl = StreamingDataLoaderBuilder::<B>::new(batcher.clone())
            .with_strategy(strategy.clone().with_shuffle(shuffle_seed))
            .with_device(device)
            .build(ds.train());

        let pbar = ProgressBar::new(dl.num_items() as u64);
        let start = Instant::now();
        for _df in dl.iter() {
            pbar.inc(batch_size as u64);
        }
        let duration = start.elapsed();
        pbar.finish_with_message("Done");
        println!("CIFAR100 train dataset preparing time: {:?}", duration);
    }

    //#[test]
    //fn test_mnist() {
    //    let mnist_path: PlRefPath =
    //        "/storage/experiments-ml/datasets/datasets--ylecun--mnist".into();

    //    let shuffle_seed = 42;
    //    let batch_size = 100;

    //    type B = Cuda;
    //    let device = CudaDevice::default();

    //    let mnist_ds = MnistDataset::new(mnist_path, crate::data::dataset::LazyFiletype::Parquet);

    //    let mnist_batcher = MnistBatcher::new();

    //    let mnist_train_dl: Arc<dyn DataLoader<B, MnistBatch<B>>> =
    //        StreamingDataLoaderBuilder::new(mnist_batcher)
    //            .with_batch_size(batch_size)
    //            .with_shuffle(shuffle_seed)
    //            .with_batch_mapper(MnistMapper::decoder())
    //            .with_device(device)
    //            .build(mnist_ds.train());

    //    let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    //    for _df in mnist_train_dl.iter() {}
    //    let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    //    println!(
    //        "Mnist train dataset preparing time: {} seconds",
    //        (end - start).as_millis()
    //    );
    //}

    #[test]
    fn test_dataset_loading() {}
}
