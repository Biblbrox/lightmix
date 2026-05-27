use crate::data::dataset::{LazyDataset, LazyFiletype};
use polars::prelude::*;

pub struct ModelNet40Dataset {}

impl LazyDataset for ModelNet40Dataset {
    fn validation(&self, path: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        let path = path.join("**/test*.*");
        self.scan(path, ft)
    }
}
