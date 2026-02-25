use std::sync::Arc;

use burn::{data::dataloader::DataLoader, prelude::*};
use polars::prelude::*;

use crate::dataloader::StreamingDataLoader;

pub mod cifar100;
pub mod imagenet1k;
pub mod mnist;

pub trait PolarsDataset {
    fn uri(&self) -> PlRefPath;

    fn train<B, O>(
        &self,
        batch_size: usize,
        shuffle_seed: Option<u64>,
        device: &B::Device,
    ) -> Arc<dyn DataLoader<B, O>>
    where
        B: Backend,
        O: From<(DataFrame, B::Device)> + Sync + Send + 'static,
    {
        let datasource = LazyFrame::scan_parquet(
            self.uri().join("**/train-*.parquet"),
            ScanArgsParquet {
                glob: true,
                ..Default::default()
            },
        )
        .unwrap();

        Arc::new(StreamingDataLoader::new(
            datasource,
            batch_size,
            shuffle_seed,
            device,
        ))
    }

    fn val<B, O>(
        &self,
        batch_size: usize,
        shuffle_seed: Option<u64>,
        device: &B::Device,
    ) -> Arc<dyn DataLoader<B, O>>
    where
        B: Backend,
        O: From<(DataFrame, B::Device)> + Sync + Send + 'static,
    {
        let datasource = LazyFrame::scan_parquet(
            self.uri().clone().join("**/validation-*.parquet"),
            ScanArgsParquet {
                glob: true,
                ..Default::default()
            },
        )
        .unwrap();

        Arc::new(StreamingDataLoader::new(
            datasource,
            batch_size,
            shuffle_seed,
            device,
        ))
    }

    fn test<B, O>(
        &self,
        batch_size: usize,
        shuffle_seed: Option<u64>,
        device: &B::Device,
    ) -> Option<Arc<dyn DataLoader<B, O>>>
    where
        B: Backend,
        O: From<(DataFrame, B::Device)> + Sync + Send + 'static,
    {
        let datasource = LazyFrame::scan_parquet(
            self.uri().clone().join("**/test-*.parquet"),
            ScanArgsParquet {
                glob: true,
                ..Default::default()
            },
        )
        .unwrap();

        Some(Arc::new(StreamingDataLoader::new(
            datasource,
            batch_size,
            shuffle_seed,
            device,
        )))
    }
}

pub fn decode<F>(column: Column, decoder: F) -> PolarsResult<Column>
where
    F: Fn(&[u8]) -> Vec<u8> + Sync + Send + 'static,
{
    let values: Vec<Vec<u8>> = column
        .struct_()?
        .field_by_name("bytes")?
        .binary()?
        .into_no_null_iter()
        .map(&decoder)
        .collect();

    Ok(Column::new("bytes".into(), values))
}

pub fn extract_imagedata(df: &DataFrame, imagecol: &str) -> PolarsResult<Vec<u8>> {
    Ok(df
        .column(imagecol)?
        .binary()?
        .into_no_null_iter()
        .flatten()
        .copied()
        .collect())
}

pub fn extract_labeldata(df: &DataFrame, labelcol: &str) -> PolarsResult<Vec<i64>> {
    Ok(df.column(labelcol)?.i64()?.into_no_null_iter().collect())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::dataset::PolarsDataset;
    use crate::dataset::imagenet1k::ImageNet1kDataset;
    use crate::dataset::{cifar100::Cifar100Batch, imagenet1k::ImageNet1kBatch, mnist::MnistBatch};
    use burn_cuda::{Cuda, CudaDevice};
    use polars::prelude::PlRefPath;

    use crate::dataset::cifar100::Cifar100Dataset;
    use crate::dataset::mnist::MnistDataset;

    #[test]
    fn test_imagenet1k() {
        let imagenet1k_path: PlRefPath =
            "/storage/experiments-ml/datasets/datasets--ILSVRC--imagenet-1k".into();

        let shuffle_seed = Some(42);
        let batch_size = 128;

        type B = Cuda;
        let device = CudaDevice::default();

        let imagenet1k_ds = ImageNet1kDataset::new(imagenet1k_path);

        let imagenet1k_train_dl =
            imagenet1k_ds.train::<B, ImageNet1kBatch<B>>(batch_size, shuffle_seed, &device);
        let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        for (idx, _df) in imagenet1k_train_dl.iter().enumerate() {
            if idx >= 9 {
                break;
            }
        }

        //for _df in imagenet1k_train_dl.iter() {}
        let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        println!(
            "ImageNet1k train dataset preparing time: {} seconds",
            (end - start).as_secs()
        );
    }

    #[test]
    fn test_cifar100() {
        let cifar100_path: PlRefPath =
            "/storage/experiments-ml/datasets/datasets--uoft-cs--cifar100".into();

        let shuffle_seed = Some(42);
        let batch_size = 128;

        type B = Cuda;
        let device = CudaDevice::default();

        let cifar100_ds = Cifar100Dataset::new(cifar100_path);

        let cifar100_train_dl =
            cifar100_ds.train::<B, Cifar100Batch<B>>(batch_size, shuffle_seed, &device);
        let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        for _df in cifar100_train_dl.iter() {}
        let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        println!(
            "CIFAR100 train dataset preparing time: {} seconds",
            (end - start).as_secs()
        );
    }

    #[test]
    fn test_mnist() {
        let mnist_path: PlRefPath =
            "/storage/experiments-ml/datasets/datasets--ylecun--mnist".into();

        let shuffle_seed = Some(42);
        let batch_size = 100;

        type B = Cuda;
        let device = CudaDevice::default();

        let mnist_ds = MnistDataset::new(mnist_path);

        let mnist_train_dl = mnist_ds.train::<B, MnistBatch<B>>(batch_size, shuffle_seed, &device);
        let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        for _df in mnist_train_dl.iter() {}
        let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        println!(
            "Mnist train dataset preparing time: {} seconds",
            (end - start).as_secs()
        );
    }

    #[test]
    fn test_dataset_loading() {}
}
