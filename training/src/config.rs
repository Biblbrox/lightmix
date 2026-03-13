use std::path::Path;

use toml::{Table, Value};

#[derive(Debug, Clone)]
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
    pub activation: String,
    pub num_encoders: i64,
    pub embed_dim: i64,
    pub num_workers: i64,
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

        Config {
            random_seed: config["random_seed"].as_integer().unwrap(),
            learning_rate: config["learning_rate"].as_float().unwrap(),
            cache_dir: config["cache_dir"].as_str().unwrap().into(),
            num_classes: config[dataset]["num_classes"].as_integer().unwrap(),
            img_size: config[dataset]["img_size"].as_integer().unwrap(),
            in_channels: config[dataset]["in_channels"].as_integer().unwrap(),
            batch_size: config[dataset][model]["batch_size"].as_integer().unwrap(),
            val_batch_size: config[dataset][model]["val_batch_size"]
                .as_integer()
                .unwrap(),
            epochs: config[dataset][model]["epochs"].as_integer().unwrap(),
            patch_size: config[dataset][model]["patch_size"].as_integer().unwrap(),
            num_heads: config[dataset][model]["num_heads"].as_integer().unwrap(),
            dropout: config[dataset][model]["dropout"].as_float().unwrap(),
            hidden_dim: config[dataset][model]["hidden_dim"].as_integer().unwrap(),
            adam_weight_decay: config[dataset][model]["adam_weight_decay"]
                .as_float()
                .unwrap(),
            adam_betas: config[dataset][model]["adam_betas"]
                .as_array()
                .unwrap()
                .as_array()
                .unwrap()
                .clone()
                .map(|v| v.as_float().unwrap()),
            activation: config[dataset][model]["activation"]
                .as_str()
                .unwrap()
                .into(),
            num_encoders: config[dataset][model]["num_encoders"].as_integer().unwrap(),
            embed_dim: config[dataset][model]["embed_dim"].as_integer().unwrap(),
            num_workers: config["num_workers"].as_integer().unwrap(),
        }
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
