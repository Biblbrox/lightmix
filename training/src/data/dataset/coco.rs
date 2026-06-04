use crate::data::dataset::{LazyDataset, LazyFiletype};

use polars::prelude::*;

pub struct CocoSegDataset {}

impl LazyDataset for CocoSegDataset {
    fn validation(&self, path: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        let path = path.join("**/test*.*");
        self.scan(path, ft)
    }
}
