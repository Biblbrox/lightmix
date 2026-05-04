use burn::prelude::Backend;
use serde::Serialize;

use crate::augmentations::colors::{ColorJitter, GaussianBlur, RandomErasing, RandomGrayscale};
use crate::augmentations::normalize::Normalize;
use crate::augmentations::rotation::{Orientation, RandomAffine, RandomFlip};
use crate::augmentations::{Augmentation, Pipeline};
use crate::config::Config;

#[derive(Debug, Clone, Serialize)]
pub struct TransformConfig {
    pub name: String,
    #[serde(default)]
    pub params: toml::Value,
}

pub struct AugmentationBuilder<B: Backend> {
    pub device: B::Device,
}

impl<B: Backend> AugmentationBuilder<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn build_from_config(&self, config: &Config) -> Pipeline<B> {
        let mut transforms: Vec<Box<dyn Augmentation<B>>> = Vec::new();

        for transform in &config.augmentations {
            match transform.name.as_str() {
                "normalize" => {
                    let std = config.std.clone();
                    let mean = config.mean.clone();
                    transforms.push(Box::new(Normalize::<B>::new(std, mean, &self.device)));
                }
                "random_flip" => {
                    let p = transform.params["probability"].as_float().unwrap();
                    let orientation = match transform.params["orientation"].as_str().unwrap() {
                        "horizontal" => Orientation::Horizontal,
                        _ => Orientation::Vertical,
                    };
                    transforms.push(Box::new(RandomFlip::<B>::new(p, orientation)));
                }
                "random_affine" => {
                    let p = transform.params["probability"].as_float().unwrap();
                    let degrees = transform.params["degrees"].as_float().unwrap();
                    transforms.push(Box::new(RandomAffine::<B>::new(p, degrees as f32)));
                }
                "color_jitter" => {
                    let brightness = transform.params["brightness"].as_float().unwrap();
                    let contrast = transform.params["contrast"].as_float().unwrap();
                    let saturation = transform.params["saturation"].as_float().unwrap();
                    transforms.push(Box::new(ColorJitter::<B>::new(
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
                    transforms.push(Box::new(er));
                }
                "gaussian_blur" => {
                    let p = transform.params["probability"].as_float().unwrap();
                    let kernel_size =
                        transform.params["kernel_size"].as_integer().unwrap() as usize;
                    let min_sigma = transform.params["min_sigma"].as_float().unwrap();
                    let max_sigma = transform.params["max_sigma"].as_float().unwrap();
                    let mut gb = GaussianBlur::<B>::new(kernel_size, &self.device);
                    gb = gb.with_p(p).with_sigma(min_sigma, max_sigma);
                    transforms.push(Box::new(gb));
                }
                "random_grayscale" => {
                    let p = transform.params["probability"].as_float().unwrap();
                    let gs = RandomGrayscale::<B>::new(p);
                    transforms.push(Box::new(gs));
                }
                _ => {
                    eprintln!("Unknown augmentation: {}", transform.name);
                }
            }
        }

        Pipeline::new(transforms)
    }
}
