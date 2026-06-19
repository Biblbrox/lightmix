use crate::data::batch::{
    Batcher, Cifar10Batcher, Cifar100Batcher, FashionMnistBatcher, Food101Batcher,
    ImageNet1kBatcher, MnistBatcher, TinyImageNetBatcher, modelnet40::ModelNet40Batcher,
};

use super::{
    LazyDataset, LazyFiletype, cifar10::Cifar10Dataset, cifar100::Cifar100Dataset,
    fashionmnist::FashionMnistDataset, food101::Food101Dataset, imagenet1k::ImageNet1kDataset,
    mnist::MnistDataset, modelnet40::ModelNet40Dataset, tinyimagenet::TinyImageNetDataset,
};
use burn::tensor::backend::Backend;
use polars::prelude::*;
use std::{str::FromStr, sync::Arc};

pub enum DatasetType {
    Cifar10,
    Cifar100,
    Mnist,
    FashionMnist,
    Food101,
    TinyImageNet,
    ImageNet1k,
    ModelNet40,
}

impl FromStr for DatasetType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cifar10" => Ok(DatasetType::Cifar10),
            "cifar100" => Ok(DatasetType::Cifar100),
            "mnist" => Ok(DatasetType::Mnist),
            "fashionmnist" => Ok(DatasetType::FashionMnist),
            "food101" => Ok(DatasetType::Food101),
            "tinyimagenet" => Ok(DatasetType::TinyImageNet),
            "imagenet1k" => Ok(DatasetType::ImageNet1k),
            "modelnet40" => Ok(DatasetType::ModelNet40),
            _ => Err(format!("Unknown dataset: {}", s)),
        }
    }
}

impl DatasetType {
    pub fn make_dataset(&self) -> DynDataset {
        match self {
            Self::Cifar10 => DynDataset(Box::new(Cifar10Dataset {})),
            Self::Cifar100 => DynDataset(Box::new(Cifar100Dataset {})),
            Self::Mnist => DynDataset(Box::new(MnistDataset {})),
            Self::FashionMnist => DynDataset(Box::new(FashionMnistDataset {})),
            Self::Food101 => DynDataset(Box::new(Food101Dataset {})),
            Self::TinyImageNet => DynDataset(Box::new(TinyImageNetDataset {})),
            Self::ImageNet1k => DynDataset(Box::new(ImageNet1kDataset {})),
            Self::ModelNet40 => DynDataset(Box::new(ModelNet40Dataset {})),
        }
    }

    pub fn make_batcher<B: Backend>(&self) -> Arc<dyn Batcher<B>> {
        match self {
            Self::Cifar10 => Cifar10Batcher::new(),
            Self::Cifar100 => Cifar100Batcher::new(),
            Self::Mnist => MnistBatcher::new(),
            Self::FashionMnist => FashionMnistBatcher::new(),
            Self::Food101 => Food101Batcher::new(),
            Self::TinyImageNet => TinyImageNetBatcher::new(),
            Self::ImageNet1k => ImageNet1kBatcher::new(),
            Self::ModelNet40 => ModelNet40Batcher::new(),
        }
    }
}

pub struct DynDataset(pub Box<dyn LazyDataset>);

impl LazyDataset for DynDataset {
    fn scan(&self, path: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        self.0.scan(path, ft)
    }

    fn train(&self, uri: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        self.0.train(uri, ft)
    }

    fn test(&self, uri: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        self.0.test(uri, ft)
    }

    fn validation(&self, uri: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        self.0.validation(uri, ft)
    }
}
