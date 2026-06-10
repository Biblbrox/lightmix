use polars::{lazy::frame::LazyFrame, prelude::PlRefPath};

use crate::data::dataset::{LazyDataset, LazyFiletype};

pub struct TinyImageNetDataset {}

impl LazyDataset for TinyImageNetDataset {
    fn validation(&self, uri: PlRefPath, ft: LazyFiletype) -> LazyFrame {
        let path = uri.join("**/val*.*");
        self.scan(path, ft)
    }
}
