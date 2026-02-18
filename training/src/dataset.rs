use burn::prelude::Backend;

use crate::dataloader::StreamingDataLoader;

mod cifar100;
mod imagenet1k;
mod mnist;

pub trait StreamableDataset<B: Backend, O> {
    fn train(&self, batch_size: usize, shuffle: bool) -> StreamingDataLoader<B, O>;
    fn val(&self, batch_size: usize, shuffle: bool) -> StreamingDataLoader<B, O>;
    fn test(&self, batch_size: usize, shuffle: bool) -> Option<StreamingDataLoader<B, O>>;
}

#[cfg(test)]
mod tests {
    use crate::dataset::StreamableDataset;
    use burn::{backend::Wgpu, data::dataloader::DataLoader};
    use polars::{frame::DataFrame, prelude::PlRefPath};

    use crate::dataset::{
        cifar100::Cifar100Dataset, imagenet1k::ImageNet1kDataset, mnist::MnistDataset,
    };

    #[test]
    fn test_dataset_loading() {
        let cache_dir: PlRefPath = "/home/iarsh/.cache/huggingface/hub".into();
        let mnist_path = cache_dir.join("datasets--ylecun--mnist");
        let imagenet1k_path = cache_dir.join("datasets--ILSVRC--imagenet-1k");
        let cifar100_path: PlRefPath = "hf://datasets/uoft-cs/cifar100".into();

        let batch_size = 100;
        let shuffle = false;

        let mnist_ds: MnistDataset<Wgpu, DataFrame> =
            MnistDataset::new(mnist_path, Default::default());
        let cifar100_ds: Cifar100Dataset<Wgpu, DataFrame> =
            Cifar100Dataset::new(cifar100_path, Default::default());
        let imagenet1k_ds: ImageNet1kDataset<Wgpu, DataFrame> =
            ImageNet1kDataset::new(imagenet1k_path, Default::default());

        let mnist_train_dl = mnist_ds.train(batch_size, shuffle);
        let cifar100_train_dl = cifar100_ds.train(batch_size, shuffle);
        let imagenet1k_val_dl = imagenet1k_ds.val(batch_size, shuffle);

        for df in mnist_train_dl.iter() {
            println!("{}", df);
        }

        for df in cifar100_train_dl.iter() {
            println!("{}", df);
        }

        for df in imagenet1k_val_dl.iter() {
            println!("{}", df);
        }
    }
}
