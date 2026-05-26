use crate::data::dataset::{LazyDataset, LazyFiletype};

use polars::prelude::*;

pub struct Cifar100Dataset {}

impl LazyDataset for Cifar100Dataset {
    fn validation(&self, path: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        let path = path.join("**/test*.*");
        self.scan(path, ft)
    }
}
