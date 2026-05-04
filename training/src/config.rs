use std::{fs::File, io::Write, path::Path};

use serde::Serialize;
use toml::{Table, Value};

use crate::augmentations::builder::TransformConfig;

#[derive(Debug, Clone, Serialize)]
pub struct Config {
    pub random_seed: i64,
    pub learning_rate: f64,
    pub cache_dir: String,
    pub num_classes: i64,
    pub img_size: i64,
    pub in_channels: i64,
    pub batch_size: i64,
    pub val_batch_size: i64,
    pub epochs: i64,
    pub patch_size: i64,
    pub num_heads: i64,
    pub dropout: f64,
    pub hidden_dim: i64,
    pub adam_weight_decay: f64,
    pub adam_betas: [f64; 2],
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub activation: String,
    pub num_encoders: i64,
    pub embed_dim: i64,
    pub num_workers: i64,
    pub continue_training: bool,
    pub resume_epoch: i64,
    pub sinkhorn_temp: f64,
    pub kernel_size: Option<i64>,
    pub augmentations: Vec<TransformConfig>,
}

impl Config {
    pub fn parse(
        config_path: &Path,
        dataset: &str,
        model: &str,
        local_config_path: Option<&Path>,
    ) -> Self {
        let file = std::fs::read_to_string(config_path).unwrap();
        let mut config: Table = file.parse().unwrap();

        if let Some(path) = local_config_path {
            let file = std::fs::read_to_string(path).unwrap();
            let localconfig: Table = file.parse().unwrap();
            Config::override_conf(&mut config, &localconfig);
        }

        let augmentations = config
            .get("augmentations")
            .and_then(|v| v.as_table())
            .and_then(|t| t.get("transforms"))
            .and_then(|v| v.as_array())
            .map(|transforms| {
                transforms
                    .iter()
                    .filter_map(|entry| {
                        let table = entry.as_table()?;
                        let name = table.get("name")?.as_str()?.to_string();

                        // Collect all fields except "name" into params
                        let params: toml::map::Map<String, toml::Value> = table
                            .iter()
                            .filter(|(k, _)| k.as_str() != "name")
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();

                        Some(TransformConfig {
                            name,
                            params: toml::Value::Table(params),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Config {
            // Global params
            random_seed: config["random_seed"].as_integer().unwrap(),
            learning_rate: config["learning_rate"].as_float().unwrap(),
            cache_dir: config["cache_dir"].as_str().unwrap().into(),
            num_workers: config["num_workers"].as_integer().unwrap(),
            continue_training: config["continue_training"].as_bool().unwrap(),
            resume_epoch: config["resume_epoch"].as_integer().unwrap(),

            // Dataset params
            num_classes: config[dataset]["num_classes"].as_integer().unwrap(),
            img_size: config[dataset]["img_size"].as_integer().unwrap(),
            in_channels: config[dataset]["in_channels"].as_integer().unwrap(),
            mean: config[dataset]["mean"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_float().unwrap() as f32)
                .collect(),
            std: config[dataset]["std"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_float().unwrap() as f32)
                .collect(),
            batch_size: config[dataset]["batch_size"].as_integer().unwrap(),
            val_batch_size: config[dataset]["val_batch_size"].as_integer().unwrap(),
            epochs: config[dataset]["epochs"].as_integer().unwrap(),

            // Model params
            patch_size: config[model]["patch_size"].as_integer().unwrap(),
            num_heads: config[model]["num_heads"].as_integer().unwrap(),
            dropout: config[model]["dropout"].as_float().unwrap(),
            hidden_dim: config[model]["hidden_dim"].as_integer().unwrap(),
            adam_weight_decay: config[model]["adam_weight_decay"].as_float().unwrap(),
            adam_betas: config[model]["adam_betas"]
                .as_array()
                .unwrap()
                .as_array()
                .unwrap()
                .clone()
                .map(|v| v.as_float().unwrap()),
            activation: config[model]["activation"].as_str().unwrap().into(),
            num_encoders: config[model]["num_encoders"].as_integer().unwrap(),
            kernel_size: match config[model].get("kernel_size") {
                Some(v) => Some(v.as_integer().unwrap()),
                None => None,
            },
            embed_dim: config[model]["embed_dim"].as_integer().unwrap(),
            sinkhorn_temp: config[model]["sinkhorn_temp"].as_float().unwrap(),
            augmentations,
        }
    }

    pub fn save(&self, path: &Path) {
        let mut file = File::create(path).unwrap();

        let config = toml::to_string_pretty(self);
        file.write_all(config.unwrap().as_bytes()).unwrap();
    }

    fn override_conf(mainconf: &mut Table, localconf: &Table) {
        for (localkey, localvalue) in localconf {
            match mainconf.get_mut(localkey) {
                Some(mainvalue) => {
                    if let (Value::Table(localtable), Value::Table(maintable)) =
                        (localvalue, mainvalue)
                    {
                        Config::override_conf(maintable, localtable);
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
}

#[cfg(test)]
mod tests {
    use std::env::current_dir;

    use crate::config::Config;

    #[test]
    fn test_parse() {
        let cwd = current_dir().unwrap();
        let path = cwd.join("experiments.toml");
        let conf = Config::parse(&path, "mnist", "model", None);

        println!("{:?}", conf);
    }

    #[test]
    fn test_parse_override() {
        let cwd = current_dir().unwrap();
        let path = cwd.join("experiments.toml");
        let localpath = cwd.join("experiments.local.toml");
        let conf = Config::parse(&path, "mnist", "model", Some(&localpath));

        assert_eq!(conf.batch_size, 8);
        println!("{:?}", conf);
    }
}
