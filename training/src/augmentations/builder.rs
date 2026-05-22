use burn::backend::Autodiff;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

use crate::augmentations::colors::{ColorJitter, GaussianBlur, RandomErasing, RandomGrayscale};
use crate::augmentations::normalize::Normalize;
use crate::augmentations::rotation::{Orientation, RandomAffine, RandomFlip};
use crate::augmentations::{Augmentation, Pipeline};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformConfig {
    pub name: String,
    #[serde(flatten)]
    pub params: std::collections::HashMap<String, toml::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AugmentationConfig {
    #[serde(default)]
    pub transforms_train: Vec<TransformConfig>,
    #[serde(default)]
    pub transforms_val: Vec<TransformConfig>,
}

pub struct AugmentationBuilder {}

impl AugmentationBuilder {
    pub fn new() -> AugmentationBuilder {
        AugmentationBuilder {}
    }

    fn match_transform<B: Backend>(
        &self,
        transform: &TransformConfig,
        transforms_vec: &mut Vec<Box<dyn Augmentation<B>>>,
        mean: &Vec<f32>,
        std: &Vec<f32>,
        device: &B::Device,
    ) {
        match transform.name.as_str() {
            "normalize" => {
                transforms_vec.push(Box::new(Normalize::<B>::new(
                    std.clone(),
                    mean.clone(),
                    device,
                )));
            }
            "random_flip" => {
                let p = transform.params["probability"].as_float().unwrap();
                let orientation = match transform.params["orientation"].as_str().unwrap() {
                    "horizontal" => Orientation::Horizontal,
                    _ => Orientation::Vertical,
                };
                transforms_vec.push(Box::new(RandomFlip::<B>::new(p, orientation)));
            }
            "random_affine" => {
                let p = transform.params["probability"].as_float().unwrap();
                let degrees = transform.params["degrees"].as_float().unwrap();
                transforms_vec.push(Box::new(RandomAffine::<B>::new(p, degrees as f32)));
            }
            "color_jitter" => {
                let brightness = transform.params["brightness"].as_float().unwrap();
                let contrast = transform.params["contrast"].as_float().unwrap();
                let saturation = transform.params["saturation"].as_float().unwrap();
                transforms_vec.push(Box::new(ColorJitter::<B>::new(
                    brightness as f32,
                    contrast as f32,
                    saturation as f32,
                )));
            }
            "random_erasing" => {
                let p = transform.params["probability"].as_float().unwrap();
                let min_scale = transform.params["min_scale"].as_float().unwrap();
                let max_scale = transform.params["max_scale"].as_float().unwrap();
                let mut er = RandomErasing::<B>::new();
                er = er.with_p(p).with_scale(min_scale, max_scale);
                transforms_vec.push(Box::new(er));
            }
            "gaussian_blur" => {
                let p = transform.params["probability"].as_float().unwrap();
                let kernel_size = transform.params["kernel_size"].as_integer().unwrap() as usize;
                let min_sigma = transform.params["min_sigma"].as_float().unwrap();
                let max_sigma = transform.params["max_sigma"].as_float().unwrap();
                let mut gb = GaussianBlur::<B>::new(kernel_size, device);
                gb = gb.with_p(p).with_sigma(min_sigma, max_sigma);
                transforms_vec.push(Box::new(gb));
            }
            "random_grayscale" => {
                let p = transform.params["probability"].as_float().unwrap();
                transforms_vec.push(Box::new(RandomGrayscale::<B>::new(p)));
            }
            _ => {
                eprintln!("Unknown augmentation: {}", transform.name);
            }
        }
    }

    pub fn build<B: Backend>(
        &self,
        config: &AugmentationConfig,
        mean: Vec<f32>,
        std: Vec<f32>,
        device: &B::Device,
    ) -> (Pipeline<Autodiff<B>>, Pipeline<B>) {
        let mut transforms_train: Vec<Box<dyn Augmentation<Autodiff<B>>>> = Vec::new();
        let mut transforms_val: Vec<Box<dyn Augmentation<B>>> = Vec::new();

        for transform in &config.transforms_train {
            self.match_transform::<Autodiff<B>>(
                transform,
                &mut transforms_train,
                &mean,
                &std,
                device,
            );
        }

        for transform in &config.transforms_val {
            self.match_transform::<B>(transform, &mut transforms_val, &mean, &std, device);
        }

        (
            Pipeline::new(transforms_train),
            Pipeline::new(transforms_val),
        )
    }
}
