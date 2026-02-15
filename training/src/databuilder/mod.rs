mod dataloader;

/// TODO: Implement a data building pipeline
struct StreamingDataLoaderBuilder;

mod schema_snippet {
    use polars::prelude::{DataType, Field, Schema};

    pub enum LocalDataset {
        MNIST,
        CIFAR100,
        ImageNet1k,
    }

    impl From<LocalDataset> for Schema {
        fn from(ds: LocalDataset) -> Self {
            match ds {
                LocalDataset::ImageNet1k => Schema::from_iter(vec![
                    Field::new(
                        "image".into(),
                        DataType::Struct(vec![
                            Field::new("bytes".into(), DataType::Binary),
                            Field::new("path".into(), DataType::String),
                        ]),
                    ),
                    Field::new("label".into(), DataType::Int64),
                ]),
                _ => Default::default(),
            }
        }
    }
}

mod datapath_scnippet {

    use std::{collections::HashMap, path::Path};

    use walkdir::{DirEntry, WalkDir};

    #[derive(Debug, PartialEq, Eq, Hash)]
    pub enum DataSplit {
        Train,
        Validation,
        Test,
    }

    /// This struct owns collections of directory entries mapped to each split
    #[derive(Debug)]
    pub struct DataPaths {
        paths: HashMap<DataSplit, Vec<DirEntry>>,
    }

    impl DataPaths {
        /// Allocate the split pathmap along with underlying vectors
        pub fn new() -> Self {
            let mut paths = HashMap::<DataSplit, Vec<DirEntry>>::with_capacity(3);
            paths.insert(DataSplit::Train, Vec::new());
            paths.insert(DataSplit::Validation, Vec::new());
            paths.insert(DataSplit::Test, Vec::new());

            Self { paths }
        }

        /// Allocate the split pathmap along with underlying vectors with specified capacities
        pub fn with_capacity(
            train_capacity: usize,
            val_capacity: usize,
            test_capacity: usize,
        ) -> Self {
            let mut paths = HashMap::<DataSplit, Vec<DirEntry>>::with_capacity(3);
            paths.insert(DataSplit::Train, Vec::with_capacity(train_capacity));
            paths.insert(DataSplit::Validation, Vec::with_capacity(val_capacity));
            paths.insert(DataSplit::Test, Vec::with_capacity(test_capacity));

            Self { paths }
        }

        /// Push path to the split collection
        pub fn append(&mut self, split: DataSplit, path: DirEntry) {
            self.paths.get_mut(&split).unwrap().push(path);
        }

        /// Get an immutable reference to the split datapaths
        pub fn get_split(&self, split: DataSplit) -> &Vec<DirEntry> {
            self.paths[&split].as_ref()
        }

        /// Walks the `data_dir` and collects parquet files corresponding to each split.
        /// Basically a QoL wrapper for `new`
        pub fn fetch(data_dir: &Path, depth: usize) -> Self {
            let mut dp = DataPaths::new();
            for entry in WalkDir::new(data_dir)
                .max_depth(depth)
                .into_iter()
                .filter(|entry| entry.as_ref().is_ok_and(is_datafile))
                .flatten()
            {
                match entry.file_name().to_str() {
                    Some(s) if s.starts_with("train") => dp.append(DataSplit::Train, entry),
                    Some(s) if s.starts_with("validation") => {
                        dp.append(DataSplit::Validation, entry)
                    }
                    Some(s) if s.starts_with("test") => dp.append(DataSplit::Test, entry),
                    _ => {}
                }
            }

            dp
        }
    }

    impl Default for DataPaths {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Parquet file filter
    fn is_datafile(entry: &DirEntry) -> bool {
        entry
            .file_name()
            .to_str()
            .is_some_and(|s| s.ends_with(".parquet"))
    }
}

mod main_snippet {

    use std::{path::Path, sync::Arc};

    use burn::{backend::Wgpu, data::dataloader::DataLoader, tensor::TensorData};
    use polars::prelude::*;

    use crate::databuilder::{
        dataloader::StreamingDataLoader,
        datapath_scnippet::{DataPaths, DataSplit},
        schema_snippet::LocalDataset,
    };
    struct ImageNet1k {
        df: DataFrame,
    }

    impl From<DataFrame> for ImageNet1k {
        fn from(value: DataFrame) -> Self {
            Self { df: value }
        }
    }

    struct LocalSource {
        lf: LazyFrame,
    }

    impl From<LocalSource> for LazyFrame {
        fn from(val: LocalSource) -> Self {
            val.lf
        }
    }

    // impl Into<TensorData> for ImageNet1k {
    //     fn into(self) -> TensorData {
    //         todo!()
    //     }
    // }

    fn main() {
        let data_dir =
            Path::new("/home/iarsh/.cache/huggingface/hub/datasets--ILSVRC--imagenet-1k/");
        let dp = DataPaths::fetch(data_dir, 4);
        let scan_args = ScanArgsParquet {
            schema: Some(Arc::new(Schema::from(LocalDataset::ImageNet1k))),
            ..Default::default()
        };
        let q = LazyFrame::scan_parquet_files(
            dp.get_split(DataSplit::Validation)
                .iter()
                .map(|e| PlRefPath::try_from_path(e.path()).unwrap())
                .collect(),
            scan_args,
        )
        .unwrap();

        println!("{}", q.explain(true).unwrap());

        let ls = LocalSource { lf: q };

        let dataloader = StreamingDataLoader::<Wgpu<f32, i32>, LocalSource, ImageNet1k>::new(
            ls,
            32,
            false,
            Default::default(),
        );
        for batch in dataloader.iter() {
            println!("{}", batch.df);
        }
    }
}
