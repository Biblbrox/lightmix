use std::str::FromStr;

use crate::models::fast_vit::{FastViT, FastViTConfig};

pub enum ModelType {
    FastViT,
    ViT,
    EfficientViT,
    FastViT3D,
}

impl FromStr for ModelType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fast_vit" => Ok(ModelType::FastViT),
            "vit" => Ok(ModelType::ViT),
            "efficient_vit" => Ok(ModelType::EfficientViT),
            "fast_vit_3d" => Ok(ModelType::FastViT3D),
            _ => Err(format!("Unknown dataset: {}", s)),
        }
    }
}

impl ModelType {
    pub fn make_model(&self) -> DynDataset {
        match self {
            Self::FastViT => DynDataset(Box::new(FastViTConfig {})),
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
