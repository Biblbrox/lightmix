use crate::data::dataset::{LazyDataset, LazyFiletype};

use polars::prelude::*;

pub struct FashionMnistDataset {}

impl LazyDataset for FashionMnistDataset {
    fn validation(&self, path: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        let path = path.join("**/test*.*");
        self.scan(path, ft)
    }
}
