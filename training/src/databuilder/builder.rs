use burn::prelude::Backend;
use polars::prelude::*;

use crate::databuilder::dataloader::StreamingDataLoader;

pub enum DataSet {
    Mnist(String),
    Cifar100(String),
    ImageNet1k(String),
}

pub enum DataSplit {
    Train,
    Validate,
    Test,
}

pub enum DataFiletype {
    Parquet,
    Arrow,
}

pub fn dataloader_builder<B: Backend, O>(
    cache_dir: PlRefPath,
    ds: &DataSet,
    split: DataSplit,
    ft: &DataFiletype,
    batch_size: usize,
    shuffle: bool,
) -> StreamingDataLoader<B, O> {
    let dsdir = match ds {
        DataSet::Mnist(p) | DataSet::Cifar100(p) | DataSet::ImageNet1k(p) => p,
    };

    let prefix = match split {
        DataSplit::Train => "train-*",
        DataSplit::Validate => "validate-*",
        DataSplit::Test => "test-*",
    };

    let fullpath = cache_dir.join(dsdir.as_str()).join("**").join(prefix);

    let query = match ft {
        DataFiletype::Parquet => LazyFrame::scan_parquet(
            fullpath,
            ScanArgsParquet {
                glob: true,
                ..Default::default()
            },
        ),
        DataFiletype::Arrow => LazyFrame::scan_ipc(
            fullpath,
            Default::default(),
            UnifiedScanArgs {
                glob: true,
                ..Default::default()
            },
        ),
    };

    StreamingDataLoader::<B, O>::new(query.unwrap(), batch_size, shuffle, B::Device::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{backend::Wgpu, data::dataloader::DataLoader};

    #[test]
    fn test_dl_builder() {
        let cache_dir: PlRefPath = "/home/iarsh/.cache/huggingface/hub".into();
        let ds = DataSet::Mnist("datasets--ylecun--mnist".into());
        let ft = DataFiletype::Parquet;

        let batch_size = 32;
        let shuffle = false;

        let train_dl: StreamingDataLoader<Wgpu, DataFrame> = dataloader_builder(
            cache_dir.clone(),
            &ds,
            DataSplit::Train,
            &ft,
            batch_size,
            shuffle,
        );
        let test_dl: StreamingDataLoader<Wgpu, DataFrame> =
            dataloader_builder(cache_dir, &ds, DataSplit::Test, &ft, batch_size, shuffle);

        for df in train_dl.iter() {
            println!("{}", df);
        }

        for df in test_dl.iter() {
            println!("{}", df);
        }
    }
}
