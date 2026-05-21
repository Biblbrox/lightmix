use std::path::Path;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use toml::{Table, Value};

use crate::augmentations::builder::AugmentationConfig;

// shared across all experiments
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
    pub augmentations: AugmentationConfig,
}

// per-dataset section
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
    pub model_table: toml::Table, // raw, to be deserialized into the concrete model type
}

fn override_conf(mainconf: &mut Table, localconf: &Table) {
    for (localkey, localvalue) in localconf {
        match mainconf.get_mut(localkey) {
            Some(mainvalue) => {
                if let (Value::Table(localtable), Value::Table(maintable)) = (localvalue, mainvalue)
                {
                    override_conf(maintable, localtable); // just call itself directly
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

    pub fn parse(path: &Path, local: Option<&Path>) -> Self {
        let file = std::fs::read_to_string(path).unwrap();
        let mut table: toml::Table = file.parse().unwrap();

        if let Some(local_path) = local {
            let local_file = std::fs::read_to_string(local_path).unwrap();
            let local_table: toml::Table = local_file.parse().unwrap();
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

[[augmentations.transforms]]
name = "random_flip"
probability = 0.5
orientation = "horizontal"

[[augmentations.transforms]]
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
        let transforms = &shared.augmentations.transforms;

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

        // Overridden values
        assert_eq!(dataset.batch_size, 8);
        assert_eq!(dataset.epochs, 10);
        assert_eq!(model.dropout, 0.2);
        assert_eq!(model.hidden_dim, 128);

        // Unchanged values
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

        // Overridden
        assert_eq!(dataset.batch_size, 16);
        assert_eq!(model.num_heads, 4);

        // Unchanged
        assert_eq!(dataset.num_classes, 10);
        assert_eq!(model.embed_dim, 512);
    }
}
