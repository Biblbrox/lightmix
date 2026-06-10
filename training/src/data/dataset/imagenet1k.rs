use polars::prelude::{LazyFrame, PlRefPath};

use crate::data::dataset::{LazyDataset, LazyFiletype};

pub struct ImageNet1kDataset {}

impl LazyDataset for ImageNet1kDataset {
    fn validation(&self, uri: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        let path = uri.join("**/val*.*");
        self.scan(path, ft)
    }
}
