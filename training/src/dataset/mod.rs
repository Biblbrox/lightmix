use std::sync::Arc;

use burn::{data::dataloader::DataLoader, prelude::*};

use crate::dataloader::StreamingDataLoader;

pub mod cifar100;
//pub mod imagenet1k;
pub mod mnist;

pub trait StreamableDataset<B: Backend, O> {
    fn train(&self, batch_size: usize, shuffle: bool) -> Arc<dyn DataLoader<B, O>>;
    fn val(&self, batch_size: usize, shuffle: bool) -> Arc<dyn DataLoader<B, O>>;
    fn test(&self, batch_size: usize, shuffle: bool) -> Option<Arc<dyn DataLoader<B, O>>>;
}

fn convert_vecs<T, U>(v: Vec<T>) -> Vec<U>
where
    T: Into<U>,
{
    v.into_iter().map(Into::into).collect()
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::dataset::{StreamableDataset, cifar100::Cifar100Batch, mnist::MnistBatch};
    use burn::{backend::Wgpu, data::dataloader::DataLoader};
    use polars::{frame::DataFrame, prelude::PlRefPath};

    use crate::dataset::cifar100::Cifar100Dataset;
    use crate::dataset::mnist::MnistDataset;

    #[test]
    fn test_dataset_loading() {
        let cache_dir: PlRefPath = "/home/biblbrox/.cache/huggingface/hub".into();
        let mnist_path: PlRefPath = "hf://datasets/ylecun/mnist".into();
        let cifar100_path: PlRefPath = "hf://datasets/uoft-cs/cifar100".into();

        let shuffle = false;
        let batch_size = 100;

        let mnist_ds: MnistDataset<Wgpu, MnistBatch<Wgpu>> =
            MnistDataset::new(mnist_path, Default::default());
        let cifar100_ds: Cifar100Dataset<Wgpu, Cifar100Batch<Wgpu>> =
            Cifar100Dataset::new(cifar100_path, Default::default());

        let mnist_train_dl = mnist_ds.train(batch_size, shuffle);
        let cifar100_train_dl = cifar100_ds.train(batch_size, shuffle);
        let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        for df in mnist_train_dl.iter() {
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
        for df in cifar100_train_dl.iter() {
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
