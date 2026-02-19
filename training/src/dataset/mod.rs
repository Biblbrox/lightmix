use std::sync::Arc;

use burn::{data::dataloader::DataLoader, prelude::Backend};
use polars::frame::DataFrame;

pub mod cifar100;
pub mod imagenet1k;
pub mod mnist;

pub trait PolarsDataset {
    fn train<B, O>(
        &self,
        batch_size: usize,
        shuffle_seed: Option<u64>,
        device: &B::Device,
    ) -> Arc<dyn DataLoader<B, O>>
    where
        B: Backend,
        O: From<(DataFrame, B::Device)> + Sync + Send + 'static;

    fn val<B, O>(
        &self,
        batch_size: usize,
        shuffle_seed: Option<u64>,
        device: &B::Device,
    ) -> Arc<dyn DataLoader<B, O>>
    where
        B: Backend,
        O: From<(DataFrame, B::Device)> + Sync + Send + 'static;

    fn test<B, O>(
        &self,
        batch_size: usize,
        shuffle_seed: Option<u64>,
        device: &B::Device,
    ) -> Option<Arc<dyn DataLoader<B, O>>>
    where
        B: Backend,
        O: From<(DataFrame, B::Device)> + Sync + Send + 'static;
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::dataset::PolarsDataset;
    use crate::dataset::{cifar100::Cifar100Batch, mnist::MnistBatch};
    use burn_cuda::{Cuda, CudaDevice};
    use polars::prelude::PlRefPath;

    use crate::dataset::cifar100::Cifar100Dataset;
    use crate::dataset::mnist::MnistDataset;

    #[test]
    fn test_dataset_loading() {
        let _cache_dir: PlRefPath = "/home/biblbrox/.cache/huggingface/hub".into();
        let mnist_path: PlRefPath = "hf://datasets/ylecun/mnist".into();
        let cifar100_path: PlRefPath = "hf://datasets/uoft-cs/cifar100".into();

        let shuffle_seed = Some(42);
        let batch_size = 100;

        type B = Cuda;
        let device = CudaDevice::default();

        let mnist_ds = MnistDataset::new(mnist_path);
        let cifar100_ds = Cifar100Dataset::new(cifar100_path);

        let mnist_train_dl = mnist_ds.train::<B, MnistBatch<B>>(batch_size, shuffle_seed, &device);
        let cifar100_train_dl =
            cifar100_ds.train::<B, Cifar100Batch<B>>(batch_size, shuffle_seed, &device);
        let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        for _df in mnist_train_dl.iter() {
            //println!(
            //    "Image shape: {}. Targets.shape: {}",
            //    df.images.shape(),
            //    df.targets.shape()
            //);
        }
        let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        println!(
            "Mnist train dataset preparing time: {} seconds",
            (end - start).as_secs()
        );
        let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        for _df in cifar100_train_dl.iter() {
            //println!(
            //    "Image shape: {}. Targets.shape: {}",
            //    df.images.shape(),
            //    df.fine_targets.shape()
            //);
        }
        let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        println!(
            "CIFAR100 train dataset preparing time: {} seconds",
            (end - start).as_secs()
        );
    }
}
