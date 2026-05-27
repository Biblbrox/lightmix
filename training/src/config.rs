use std::path::Path;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use toml::{Table, Value};

use crate::augmentations::builder::AugmentationConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedConfig {
    pub random_seed: i64,
    pub learning_rate: f64,
    pub cache_dir: String,
    pub num_workers: i64,
    pub continue_training: bool,
    pub resume_epoch: i64,
    pub active_dataset: String,
    pub active_model: String,
    #[serde(default)]
    pub augmentations: AugmentationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub num_classes: usize,
    pub img_size: usize,
    pub in_channels: usize,
    pub batch_size: usize,
    pub val_batch_size: usize,
    pub epochs: usize,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
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
        let mut dataset_name = shared.active_dataset.clone();
        let mut model_name = shared.active_model.clone();

        let mut dataset_section = datasets_table[&dataset_name].as_table().cloned().unwrap();
        let mut model_section = models_table[&model_name].as_table().cloned().unwrap();

        // Load augmentations (not in experiments.toml, so attach after parsing shared)
        let aug_config = Self::parse_augmentations(config_dir.join("augmentations.toml"));
        shared.augmentations = aug_config;

        match local {
            Some(local_path) if local_path.exists() => {
                let local_table = Self::read_toml(local_path);

                // Collect shared fields from local top-level keys
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
                            dataset_section = datasets_table[new_ds].as_table().cloned().unwrap();
                        }
                        "active_model" => {
                            let new_md = value.as_str().unwrap();
                            model_name = new_md.to_string();
                            model_section = models_table[new_md].as_table().cloned().unwrap();
                        }
                        _ => {}
                    }
                }

                // Re-parse shared after merging local overrides
                if !local_shared_keys.is_empty() {
                    let mut local_shared = Table::new();
                    for (key, value) in local_shared_keys {
                        local_shared.insert(key, (*value).clone());
                    }
                    let mut merged_shared = shared_table.clone();
                    override_conf(&mut merged_shared, &local_shared);
                    shared = merged_shared.try_into().unwrap();
                    // Re-attach augmentations (not in experiments.toml anymore)
                    let aug_config =
                        Self::parse_augmentations(config_dir.join("augmentations.toml"));
                    shared.augmentations = aug_config;
                }

                // Merge dataset section from local if present
                if let Some(ds_local) = local_table.get(&dataset_name)
                    && let Value::Table(local_ds) = ds_local
                {
                    let mut merged_ds = dataset_section.clone();
                    override_conf(&mut merged_ds, local_ds);
                    dataset_section = merged_ds;
                }

                // Merge model section from local if present
                if let Some(md_local) = local_table.get(&model_name)
                    && let Value::Table(local_md) = md_local
                {
                    let mut merged_md = model_section.clone();
                    override_conf(&mut merged_md, local_md);
                    model_section = merged_md;
                }
            }
            _ => (),
        }

        let dataset: DatasetConfig = dataset_section.try_into().unwrap();

        ParsedConfig {
            shared,
            dataset,
            model_table: model_section,
        }
    }

    /// Parse [[transforms]] array with `kind` discriminator (train/val).
    fn parse_augmentations<P: AsRef<Path>>(path: P) -> AugmentationConfig {
        let path = path.as_ref();
        let file = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(_) => {
                return AugmentationConfig {
                    transforms_train: vec![],
                    transforms_val: vec![],
                };
            }
        };

        let table: toml::Table = match file.parse() {
            Ok(t) => t,
            Err(_) => {
                return AugmentationConfig {
                    transforms_train: vec![],
                    transforms_val: vec![],
                };
            }
        };

        let mut transforms_train: Vec<crate::augmentations::builder::TransformConfig> = vec![];
        let mut transforms_val: Vec<crate::augmentations::builder::TransformConfig> = vec![];

        if let Some(transforms) = table.get("transforms").and_then(|v| v.as_array()) {
            for item in transforms {
                if let Some(table) = item.as_table() {
                    let name = table
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let kind = table.get("kind").and_then(|v| v.as_str()).unwrap_or("");

                    // Collect params (everything except name and kind)
                    let mut params: std::collections::HashMap<String, toml::Value> =
                        std::collections::HashMap::new();
                    for (k, v) in table {
                        if *k != "name" && *k != "kind" {
                            params.insert(k.clone(), v.clone());
                        }
                    }

                    let transform = crate::augmentations::builder::TransformConfig { name, params };

                    match kind {
                        "train" => transforms_train.push(transform),
                        "val" => transforms_val.push(transform),
                        _ => {}
                    }
                }
            }
        }

        AugmentationConfig {
            transforms_train,
            transforms_val,
        }
    }

    /// Legacy: parse from a single combined TOML file (backward compat with tests).
    pub fn parse(path: &Path, local: Option<&Path>) -> Self {
        let mut table: toml::Table = Self::read_toml(path);

        if let Some(local_path) = local {
            let local_table = Self::read_toml(local_path);
            override_conf(&mut table, &local_table);
        }

        let shared: SharedConfig = table.clone().try_into().unwrap();
        let dataset = table[&shared.active_dataset].clone().try_into().unwrap();
        let model_table = table[&shared.active_model].as_table().unwrap().clone();

        ParsedConfig {
            shared,
            dataset,
            model_table,
        }
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
    use tempfile::TempDir;

    fn create_test_config() -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new().unwrap();
        let config_path = dir.path().join("experiments.toml");

        let config_content = r#"
random_seed = 42
learning_rate = 0.001
cache_dir = "/tmp/cache"
num_workers = 4
continue_training = false
resume_epoch = 0
active_dataset = "mnist"
active_model = "model"

[augmentations]

[[augmentations.transforms_train]]
name = "random_flip"
probability = 0.5
orientation = "horizontal"

[[augmentations.transforms_train]]
name = "color_jitter"
brightness = 0.2
contrast = 0.2
saturation = 0.2

[mnist]
num_classes = 10
img_size = 28
in_channels = 1
mean = [0.1307]
std = [0.3081]
batch_size = 64
val_batch_size = 128
epochs = 100

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

        std::fs::write(&config_path, config_content).unwrap();
        (dir, config_path)
    }

    fn create_test_local_config() -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new().unwrap();
        let local_config_path = dir.path().join("experiments.local.toml");

        let local_config_content = r#"
[mnist]
batch_size = 8
epochs = 10

[model]
dropout = 0.2
hidden_dim = 128
"#;

        std::fs::write(&local_config_path, local_config_content).unwrap();
        (dir, local_config_path)
    }

    #[test]
    fn test_parse_basic_shared_config() {
        let (_dir, config_path) = create_test_config();
        let config = ParsedConfig::parse(&config_path, None);
        let shared: SharedConfig = config.shared;

        assert_eq!(shared.random_seed, 42);
        assert_eq!(shared.learning_rate, 0.001);
        assert_eq!(shared.cache_dir, "/tmp/cache");
        assert_eq!(shared.num_workers, 4);
        assert_eq!(shared.continue_training, false);
        assert_eq!(shared.resume_epoch, 0);
        assert_eq!(shared.active_dataset, "mnist");
        assert_eq!(shared.active_model, "model");
    }

    #[test]
    fn test_parse_basic_dataset_config() {
        let (_dir, config_path) = create_test_config();
        let config = ParsedConfig::parse(&config_path, None);
        let dataset: DatasetConfig = config.dataset;

        assert_eq!(dataset.num_classes, 10);
        assert_eq!(dataset.img_size, 28);
        assert_eq!(dataset.in_channels, 1);
        assert_eq!(dataset.mean, vec![0.1307]);
        assert_eq!(dataset.std, vec![0.3081]);
        assert_eq!(dataset.batch_size, 64);
        assert_eq!(dataset.val_batch_size, 128);
        assert_eq!(dataset.epochs, 100);
    }

    #[test]
    fn test_parse_basic_model_config() {
        let (_dir, config_path) = create_test_config();
        let config = ParsedConfig::parse(&config_path, None);
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
    fn test_parse_optimizer_config() {
        let (_dir, config_path) = create_test_config();
        let config = ParsedConfig::parse(&config_path, None);
        let optimizer: OptimizerConfig = config.optimizer();

        assert_eq!(optimizer.adam_weight_decay, 0.0001);
        assert_eq!(optimizer.adam_betas, [0.9, 0.999]);
    }

    #[test]
    fn test_parse_augmentations() {
        let (_dir, config_path) = create_test_config();
        let config = ParsedConfig::parse(&config_path, None);
        let shared: SharedConfig = config.shared;
        let transforms = &shared.augmentations.transforms_train;

        assert_eq!(transforms.len(), 2);
        assert_eq!(transforms[0].name, "random_flip");
        assert_eq!(transforms[0].params["probability"].as_float().unwrap(), 0.5);
        assert_eq!(
            transforms[0].params["orientation"].as_str().unwrap(),
            "horizontal"
        );
        assert_eq!(transforms[1].name, "color_jitter");
        assert_eq!(transforms[1].params["brightness"].as_float().unwrap(), 0.2);
    }

    #[test]
    fn test_parse_with_override() {
        let (_dir, config_path) = create_test_config();
        let (_local_dir, local_config_path) = create_test_local_config();

        let config = ParsedConfig::parse(&config_path, Some(&local_config_path));
        let (_, dataset, model_table) = config.unpack();
        let model: FastViTConfig = model_table.try_into().unwrap();

        assert_eq!(dataset.batch_size, 8);
        assert_eq!(dataset.epochs, 10);
        assert_eq!(model.dropout, 0.2);
        assert_eq!(model.hidden_dim, 128);

        assert_eq!(dataset.num_classes, 10);
        assert_eq!(model.patch_size, 7);
    }

    #[test]
    fn test_override_adds_new_fields() {
        let (_dir, config_path) = create_test_config();
        let local_dir = TempDir::new().unwrap();
        let local_config_path = local_dir.path().join("experiments.local.toml");

        let local_config_content = r#"
[mnist]
batch_size = 16

[model]
num_heads = 4
"#;
        std::fs::write(&local_config_path, local_config_content).unwrap();

        let config = ParsedConfig::parse(&config_path, Some(&local_config_path));
        let (_, dataset, model_table) = config.unpack();
        let model: FastViTConfig = model_table.try_into().unwrap();

        assert_eq!(dataset.batch_size, 16);
        assert_eq!(model.num_heads, 4);

        assert_eq!(dataset.num_classes, 10);
        assert_eq!(model.embed_dim, 512);
    }
}
