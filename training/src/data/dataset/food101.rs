use crate::data::dataset::{LazyDataset, LazyFiletype};

use polars::prelude::*;

pub struct Food101Dataset {
    uri: PlRefPath,
    ft: LazyFiletype,
    train_subpath: Option<PlRefPath>,
    val_subpath: Option<PlRefPath>,
    test_subpath: Option<PlRefPath>,
    parquet_args: Option<ScanArgsParquet>,
    arrow_args: Option<(IpcScanOptions, UnifiedScanArgs)>,
}

impl Food101Dataset {
    pub fn new(uri: impl Into<PlRefPath>, ft: LazyFiletype) -> Self {
        Self {
            uri: uri.into(),
            ft,
            train_subpath: None,
            val_subpath: None,
            test_subpath: None,
            parquet_args: None,
            arrow_args: None,
        }
    }

    pub fn with_filetype(mut self, ft: LazyFiletype) -> Self {
        self.ft = ft;
        self
    }

    pub fn with_train_subpath(mut self, train_subpath: impl Into<PlRefPath>) -> Self {
        self.train_subpath = Some(train_subpath.into());
        self
    }

    pub fn with_val_subpath(mut self, val_subpath: impl Into<PlRefPath>) -> Self {
        self.val_subpath = Some(val_subpath.into());
        self
    }

    pub fn with_test_subpath(mut self, test_subpath: impl Into<PlRefPath>) -> Self {
        self.test_subpath = Some(test_subpath.into());
        self
    }

    pub fn with_parquet_args(mut self, args: ScanArgsParquet) -> Self {
        self.parquet_args = Some(args);
        self
    }

    pub fn with_arrow_args(
        mut self,
        ipc_args: IpcScanOptions,
        unified_args: UnifiedScanArgs,
    ) -> Self {
        self.arrow_args = Some((ipc_args, unified_args));
        self
    }
}

impl LazyDataset for Food101Dataset {
    fn train(&self) -> LazyFrame {
        let path = match self.train_subpath {
            Some(ref subpath) => self.uri.join(subpath),
            None => self.uri.join("**/train*.*"),
        };

        match self.ft {
            LazyFiletype::Parquet => {
                let parquet_args = self.parquet_args.clone().unwrap_or_default();
                LazyFrame::scan_parquet(path, parquet_args).unwrap()
            }
            LazyFiletype::Arrow => {
                let (ipc_args, unified_args) = self.arrow_args.clone().unwrap_or_default();
                LazyFrame::scan_ipc(path, ipc_args, unified_args).unwrap()
            }
            _ => unimplemented!(),
        }
    }

    fn validation(&self) -> LazyFrame {
        let path = match self.val_subpath {
            Some(ref subpath) => self.uri.join(subpath),
            None => self.uri.join("**/val*.*"),
        };

        match self.ft {
            LazyFiletype::Parquet => {
                let parquet_args = self.parquet_args.clone().unwrap_or_default();
                LazyFrame::scan_parquet(path, parquet_args).unwrap()
            }
            LazyFiletype::Arrow => {
                let (ipc_args, unified_args) = self.arrow_args.clone().unwrap_or_default();
                LazyFrame::scan_ipc(path, ipc_args, unified_args).unwrap()
            }
            _ => unimplemented!(),
        }
    }

    fn test(&self) -> LazyFrame {
        let path = match self.test_subpath {
            Some(ref subpath) => self.uri.join(subpath),
            None => self.uri.join("**/test*.*"),
        };

        match self.ft {
            LazyFiletype::Parquet => {
                let parquet_args = self.parquet_args.clone().unwrap_or_default();
                LazyFrame::scan_parquet(path, parquet_args).unwrap()
            }
            LazyFiletype::Arrow => {
                let (ipc_args, unified_args) = self.arrow_args.clone().unwrap_or_default();
                LazyFrame::scan_ipc(path, ipc_args, unified_args).unwrap()
            }
            _ => unimplemented!(),
        }
    }
}
