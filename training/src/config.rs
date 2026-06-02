use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use toml::{Table, Value};

use crate::augmentations::builder::{AugmentationConfig, PipelineDefault, PipelineDefaults};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedConfig {
    pub random_seed: i64,
    pub learning_rate: f64,
    pub cache_dir: String,
    pub num_workers: i64,
    pub continue_training: bool,
    pub resume_epoch: i64,
    #[serde(default)]
    pub early_stopping: bool,
    pub active_dataset: String,
    pub active_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub num_classes: usize,
    pub img_size: usize,
    pub in_channels: usize,
    pub batch_size: usize,
    pub val_batch_size: usize,
    pub epochs: usize,
    #[serde(default = "default_dataset_type")]
    pub dataset_type: String,
    /// Resolved augmentation pipeline for this dataset.
    #[serde(default)]
    pub augmentations: AugmentationConfig,
}

fn default_dataset_type() -> String {
    "Arrow".to_string()
}

/// Temporary struct used only during config parsing (not serialized).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawDatasetConfig {
    pub num_classes: usize,
    pub img_size: usize,
    pub in_channels: usize,
    pub batch_size: usize,
    pub val_batch_size: usize,
    pub epochs: usize,
    #[serde(default)]
    pub dataset_type: String,
    #[serde(default)]
    pub augmentation_pipeline: String,
    #[serde(default)]
    pub transforms: Vec<String>,
    /// Per-transform overrides as a table: [dataset.augmentations.transform_name]
    #[serde(default)]
    pub augmentations: std::collections::HashMap<String, toml::Value>,
}

impl From<RawDatasetConfig> for DatasetConfig {
    fn from(raw: RawDatasetConfig) -> Self {
        // Placeholder — resolved fields are set by resolve_dataset_augmentations()
        DatasetConfig {
            num_classes: raw.num_classes,
            img_size: raw.img_size,
            in_channels: raw.in_channels,
            batch_size: raw.batch_size,
            val_batch_size: raw.val_batch_size,
            epochs: raw.epochs,
            dataset_type: if raw.dataset_type.is_empty() {
                "Arrow".to_string()
            } else {
                raw.dataset_type
            },
            augmentations: AugmentationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub adam_weight_decay: f64,
    pub adam_betas: [f64; 2],
}

pub struct ParsedConfig {
    pub shared: SharedConfig,
    pub dataset: DatasetConfig,
    pub model_table: toml::Table, // raw table, deserialized into concrete type at runtime
}

fn override_conf(mainconf: &mut Table, localconf: &Table) {
    for (localkey, localvalue) in localconf {
        match mainconf.get_mut(localkey) {
            Some(mainvalue) => {
                if let (Value::Table(localtable), Value::Table(maintable)) = (localvalue, mainvalue)
                {
                    override_conf(maintable, localtable);
                } else {
                    let _ = mainconf.insert(localkey.clone(), localvalue.clone());
                }
            }
            None => {
                let _ = mainconf.insert(localkey.clone(), localvalue.clone());
            }
        }
    }
}

impl ParsedConfig {
    pub fn unpack(self) -> (SharedConfig, DatasetConfig, toml::Table) {
        (self.shared, self.dataset, self.model_table)
    }

    /// Load from separate config files in the configs/ directory.
    pub fn load(config_dir: &Path, local: Option<&Path>) -> Self {
        let shared_table = Self::read_toml(config_dir.join("experiments.toml"));
        let datasets_table = Self::read_toml(config_dir.join("datasets.toml"));
        let models_table = Self::read_toml(config_dir.join("models.toml"));

        // Extract active_dataset and active_model before overrides so we can re-fetch sections
        let mut shared: SharedConfig = shared_table.clone().try_into().unwrap();
        let dataset_name = shared.active_dataset.clone();
        let model_name = shared.active_model.clone();

        let mut dataset_section = datasets_table[&dataset_name].as_table().cloned().unwrap();
        let mut model_section = models_table[&model_name].as_table().cloned().unwrap();

        // Parse pipeline defaults once (used for all dataset augmentation resolution)
        let pipelines = Self::parse_pipeline_defaults(config_dir.join("augmentations.toml"));

        if let Some(local_path) = local.filter(|p| p.exists()) {
            let local_table = Self::read_toml(local_path);
            (shared, dataset_section, model_section) = Self::apply_local_overrides(
                shared,
                &shared_table,
                &datasets_table,
                &models_table,
                local_table,
                config_dir,
                &pipelines,
            );
        }

        // Resolve augmentations from pipeline defaults + dataset overrides
        let raw_dataset: RawDatasetConfig = dataset_section.try_into().unwrap();
        let resolved = Self::resolve_dataset_augmentations(&raw_dataset, &pipelines);

        ParsedConfig {
            shared,
            dataset: DatasetConfig {
                num_classes: raw_dataset.num_classes,
                img_size: raw_dataset.img_size,
                in_channels: raw_dataset.in_channels,
                batch_size: raw_dataset.batch_size,
                val_batch_size: raw_dataset.val_batch_size,
                epochs: raw_dataset.epochs,
                dataset_type: if raw_dataset.dataset_type.is_empty() {
                    "Arrow".to_string()
                } else {
                    raw_dataset.dataset_type
                },
                augmentations: resolved,
            },
            model_table: model_section,
        }
    }

    /// Apply local config overrides: update shared fields, dataset, and model sections.
    fn apply_local_overrides(
        mut shared: SharedConfig,
        shared_table: &Table,
        datasets_table: &Table,
        models_table: &Table,
        local_table: Table,
        _config_dir: &Path,
        _pipelines: &HashMap<String, PipelineDefaults>,
    ) -> (SharedConfig, Table, Table) {
        let mut dataset_name = shared.active_dataset.clone();
        let mut model_name = shared.active_model.clone();

        // Collect shared fields and detect active_dataset/active_model changes
        let mut local_shared_keys: Vec<(String, &Value)> = vec![];
        for (key, value) in &local_table {
            match key.as_str() {
                "random_seed" | "learning_rate" | "cache_dir" | "num_workers"
                | "continue_training" | "resume_epoch" => {
                    local_shared_keys.push((key.clone(), value));
                }
                "active_dataset" => {
                    let new_ds = value.as_str().unwrap();
                    dataset_name = new_ds.to_string();
                }
                "active_model" => {
                    let new_md = value.as_str().unwrap();
                    model_name = new_md.to_string();
                }
                _ => {}
            }
        }

        // Update dataset_section and model_section after potential name changes
        let mut dataset_section = datasets_table[&dataset_name].as_table().cloned().unwrap();
        let mut model_section = models_table[&model_name].as_table().cloned().unwrap();

        // Re-parse shared after merging local overrides
        if !local_shared_keys.is_empty() {
            let mut local_shared = Table::new();
            for (key, value) in local_shared_keys {
                local_shared.insert(key, (*value).clone());
            }
            let mut merged_shared = shared_table.clone();
            override_conf(&mut merged_shared, &local_shared);
            shared = merged_shared.try_into().unwrap();
        }

        // Merge dataset section from local if present
        if let Some(ds_local) = local_table.get(&dataset_name).and_then(|v| v.as_table()) {
            override_conf(&mut dataset_section, ds_local);
        }

        // Merge model section from local if present
        if let Some(md_local) = local_table.get(&model_name).and_then(|v| v.as_table()) {
            override_conf(&mut model_section, md_local);
        }

        (shared, dataset_section, model_section)
    }

    /// Resolve a dataset's AugmentationConfig from pipeline defaults + table-based overrides.
    fn resolve_dataset_augmentations(
        raw: &RawDatasetConfig,
        pipelines: &HashMap<String, PipelineDefaults>,
    ) -> AugmentationConfig {
        if raw.augmentation_pipeline.is_empty() || raw.transforms.is_empty() {
            // No pipeline or no transforms specified — return empty config
            return AugmentationConfig::default();
        }

        // Look up the named pipeline (panic with clear error if not found)
        let pipeline = pipelines
            .get(&raw.augmentation_pipeline)
            .unwrap_or_else(|| {
                panic!(
                    "Dataset '{}' references unknown augmentation pipeline '{}'. Available: {:?}",
                    raw.num_classes,
                    raw.augmentation_pipeline,
                    pipelines.keys().collect::<Vec<_>>()
                )
            });

        // Build a lookup of default transforms by name for easy override matching
        let defaults_by_name: HashMap<&str, &PipelineDefault> = pipeline
            .transforms
            .iter()
            .map(|t| (t.name.as_str(), t))
            .collect();

        let mut transforms_train: Vec<crate::augmentations::builder::TransformConfig> = vec![];
        let mut transforms_val: Vec<crate::augmentations::builder::TransformConfig> = vec![];

        // Process transforms in the specified order, deep-merging overrides with defaults
        for transform_name in &raw.transforms {
            let default_transform = defaults_by_name
                .get(transform_name.as_str())
                .unwrap_or_else(|| {
                    panic!(
                        "Dataset '{}' orders transform '{}' but it is not defined in pipeline '{}'",
                        raw.num_classes, transform_name, raw.augmentation_pipeline
                    )
                });

            // Deep-merge override params into default params (override fills/overwrites defaults)
            let params = match raw.augmentations.get(transform_name.as_str()) {
                Some(override_table) => {
                    Self::deep_merge_params(&default_transform.params, override_table)
                }
                None => default_transform.params.clone(),
            };

            let transform = crate::augmentations::builder::TransformConfig {
                name: transform_name.clone(),
                params,
            };

            match default_transform.kind.as_str() {
                "train" => transforms_train.push(transform),
                "val" => transforms_val.push(transform),
                _ => {}
            }
        }

        // Check that all overrides reference existing transforms (catch typos)
        raw.augmentations.keys().for_each(|override_name| {
            if !defaults_by_name.contains_key(override_name.as_str()) {
                panic!(
                    "Override references unknown transform '{}' in dataset '{}'",
                    override_name, raw.num_classes
                );
            }
        });

        AugmentationConfig {
            transforms_train,
            transforms_val,
        }
    }

    /// Deep-merge override params into default params. Override values fill/overwrite defaults.
    fn deep_merge_params(
        defaults: &HashMap<String, toml::Value>,
        overrides: &toml::Value,
    ) -> HashMap<String, toml::Value> {
        let mut merged = defaults.clone();
        if let Some(override_table) = overrides.as_table() {
            for (k, v) in override_table {
                merged.insert(k.clone(), v.clone());
            }
        }
        merged
    }

    /// Parse pipeline defaults from augmentations.toml into a name → PipelineDefaults map.
    fn parse_pipeline_defaults<P: AsRef<Path>>(path: P) -> HashMap<String, PipelineDefaults> {
        let path = path.as_ref();
        let file = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(_) => return HashMap::new(),
        };

        let table: toml::Table = match file.parse() {
            Ok(t) => t,
            Err(_) => return HashMap::new(),
        };

        let mut pipelines = HashMap::new();

        if let Some(defaults_table) = table.get("defaults").and_then(|v| v.as_table()) {
            for (pipeline_name, pipeline_value) in defaults_table {
                match pipeline_value.clone().try_into::<PipelineDefaults>() {
                    Ok(pipeline_defaults) => {
                        pipelines.insert(pipeline_name.clone(), pipeline_defaults);
                    }
                    Err(_) => {
                        eprintln!("Failed to parse pipeline '{}', skipping", pipeline_name);
                    }
                }
            }
        }

        pipelines
    }

    fn read_toml<P: AsRef<Path>>(path: P) -> Table {
        let path = path.as_ref();
        match std::fs::read_to_string(path) {
            Ok(content) => content.parse().unwrap_or_else(|e| {
                panic!("Failed to parse {}: {}", path.display(), e);
            }),
            Err(e) => {
                // Return empty table for optional files (.local.toml) instead of panicking
                if path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .map(|n| n.ends_with(".local.toml"))
                    .unwrap_or(false)
                {
                    eprintln!("Local config not found at {}, skipping", path.display());
                    Table::new()
                } else {
                    panic!("Failed to read {}: {}", path.display(), e);
                }
            }
        }
    }

    pub fn model<M: DeserializeOwned>(&self) -> M {
        self.model_table.clone().try_into().unwrap()
    }

    pub fn optimizer(&self) -> OptimizerConfig {
        self.model_table.clone().try_into().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{DatasetConfig, OptimizerConfig, ParsedConfig, SharedConfig};
    use crate::models::fast_vit::FastViTConfig;
    use std::path::PathBuf;
    use tempfile::TempDir;

    /// Creates the full multi-file config directory with experiments.toml,
    /// datasets.toml, models.toml, and augmentations.toml.
    fn create_test_config() -> TempDir {
        let dir = TempDir::new().unwrap();

        // experiments.toml — shared settings + active references
        let experiments = r#"
random_seed = 42
learning_rate = 0.001
cache_dir = "/tmp/cache"
num_workers = 4
continue_training = false
resume_epoch = 0
active_dataset = "mnist"
active_model = "model"
"#;
        std::fs::write(dir.path().join("experiments.toml"), experiments).unwrap();

        // datasets.toml — dataset-specific settings + pipeline reference
        let datasets = r#"
[mnist]
num_classes = 10
img_size = 28
in_channels = 1
batch_size = 64
val_batch_size = 128
epochs = 100
dataset_type = "Arrow"
augmentation_pipeline = "image"
transforms = ["normalize", "random_flip"]

[[mnist.augmentations.normalize]]
mean = [0.1307]
std = [0.3081]

[[mnist.augmentations.random_flip]]
probability = 0.5
orientation = "horizontal"
"#;
        std::fs::write(dir.path().join("datasets.toml"), datasets).unwrap();

        // models.toml — model-specific settings + optimizer defaults
        let models = r#"
[model]
patch_size = 7
num_heads = 8
dropout = 0.1
hidden_dim = 256
adam_weight_decay = 0.0001
adam_betas = [0.9, 0.999]
activation = "gelu"
num_encoders = 6
embed_dim = 512
sinkhorn_temp = 0.1
"#;
        std::fs::write(dir.path().join("models.toml"), models).unwrap();

        // augmentations.toml — pipeline defaults (referenced by dataset)
        let augmentations = r#"
[defaults.image]
[[defaults.image.transforms]]
name = "normalize"
kind = "train"
mean = [0.5]
std = [0.5]

[[defaults.image.transforms]]
name = "random_flip"
kind = "train"
probability = 0.5
orientation = "horizontal"

[[defaults.image.transforms]]
name = "color_jitter"
kind = "train"
brightness = 0.2
contrast = 0.2
saturation = 0.2
"#;
        std::fs::write(dir.path().join("augmentations.toml"), augmentations).unwrap();

        dir
    }

    /// Creates a local override file alongside the main config directory.
    fn create_test_local_config(base_dir: &TempDir) -> PathBuf {
        let local_path = base_dir.path().join("experiments.local.toml");
        let local_content = r#"
[mnist]
batch_size = 8
epochs = 10

[model]
dropout = 0.2
hidden_dim = 128
"#;
        std::fs::write(&local_path, local_content).unwrap();
        local_path
    }

    #[test]
    fn test_load_basic_shared_config() {
        let dir = create_test_config();
        let config_dir = &dir.path();
        let config = ParsedConfig::load(config_dir, None);
        let shared: SharedConfig = config.shared;

        assert_eq!(shared.random_seed, 42);
        assert_eq!(shared.learning_rate, 0.001);
        assert_eq!(shared.cache_dir, "/tmp/cache");
        assert_eq!(shared.num_workers, 4);
        assert!(!shared.continue_training);
        assert_eq!(shared.resume_epoch, 0);
        assert_eq!(shared.active_dataset, "mnist");
        assert_eq!(shared.active_model, "model");
    }

    #[test]
    fn test_load_basic_dataset_config() {
        let dir = create_test_config();
        let config_dir = &dir.path();
        let config = ParsedConfig::load(config_dir, None);
        let dataset: DatasetConfig = config.dataset;

        assert_eq!(dataset.num_classes, 10);
        assert_eq!(dataset.img_size, 28);
        assert_eq!(dataset.in_channels, 1);
        assert_eq!(dataset.batch_size, 64);
        assert_eq!(dataset.val_batch_size, 128);
        assert_eq!(dataset.epochs, 100);
    }

    #[test]
    fn test_load_basic_model_config() {
        let dir = create_test_config();
        let config_dir = &dir.path();
        let config = ParsedConfig::load(config_dir, None);
        let model: FastViTConfig = config.model();

        assert_eq!(model.patch_size, 7);
        assert_eq!(model.num_heads, 8);
        assert_eq!(model.dropout, 0.1);
        assert_eq!(model.hidden_dim, 256);
        assert_eq!(model.activation, "gelu");
        assert_eq!(model.num_encoders, 6);
        assert_eq!(model.embed_dim, 512);
        assert_eq!(model.sinkhorn_temp, 0.1);
    }

    #[test]
    fn test_load_optimizer_config() {
        let dir = create_test_config();
        let config_dir = &dir.path();
        let config = ParsedConfig::load(config_dir, None);
        let optimizer: OptimizerConfig = config.optimizer();

        assert_eq!(optimizer.adam_weight_decay, 0.0001);
        assert_eq!(optimizer.adam_betas, [0.9, 0.999]);
    }

    #[test]
    fn test_load_augmentations_resolved() {
        let dir = create_test_config();
        let config_dir = &dir.path();
        let config = ParsedConfig::load(config_dir, None);
        let dataset: DatasetConfig = config.dataset;

        // New approach: augmentations are resolved into the dataset config,
        // not stored in SharedConfig.augmentations.
        let transforms = &dataset.augmentations;

        assert_eq!(transforms.transforms_train.len(), 2);
        assert_eq!(transforms.transforms_val.len(), 0);

        assert_eq!(transforms.transforms_train[0].name, "normalize");
        assert_eq!(transforms.transforms_train[1].name, "random_flip");
        assert_eq!(
            transforms.transforms_train[1].params["probability"]
                .as_float()
                .unwrap(),
            0.5
        );
        assert_eq!(
            transforms.transforms_train[1].params["orientation"]
                .as_str()
                .unwrap(),
            "horizontal"
        );
    }

    #[test]
    fn test_load_with_override() {
        let dir = create_test_config();
        let config_dir = &dir.path();
        let local_path = create_test_local_config(&dir);

        let config = ParsedConfig::load(config_dir, Some(&local_path));
        let (_, dataset, model_table) = config.unpack();
        let model: FastViTConfig = model_table.try_into().unwrap();

        assert_eq!(dataset.batch_size, 8);
        assert_eq!(dataset.epochs, 10);
        assert_eq!(model.dropout, 0.2);
        assert_eq!(model.hidden_dim, 128);

        // Fields not overridden should retain base values
        assert_eq!(dataset.num_classes, 10);
        assert_eq!(model.patch_size, 7);
    }

    #[test]
    fn test_override_adds_new_fields() {
        let dir = create_test_config();
        let config_dir = &dir.path();

        // Create a local file that only overrides specific fields
        let local_content = r#"
[mnist]
batch_size = 16

[model]
num_heads = 4
"#;
        let local_path = dir.path().join("experiments.local.toml");
        std::fs::write(&local_path, local_content).unwrap();

        let config = ParsedConfig::load(config_dir, Some(&local_path));
        let (_, dataset, model_table) = config.unpack();
        let model: FastViTConfig = model_table.try_into().unwrap();

        assert_eq!(dataset.batch_size, 16);
        assert_eq!(model.num_heads, 4);

        // Unchanged fields should retain base values
        assert_eq!(dataset.num_classes, 10);
        assert_eq!(model.embed_dim, 512);
    }
}
