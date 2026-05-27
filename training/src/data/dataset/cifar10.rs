use crate::data::dataset::{LazyDataset, LazyFiletype};

use polars::prelude::*;

pub struct Cifar10Dataset {}

impl LazyDataset for Cifar10Dataset {
    fn validation(&self, path: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        let path = path.join("**/test*.*");
        self.scan(path, ft)
    }
}
