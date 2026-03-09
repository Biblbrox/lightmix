use std::marker::PhantomData;

use burn::{Tensor, prelude::Backend, tensor::Shape};
use rand::RngExt;

use crate::augmentations::Augmentation;

pub struct ColorJitter<B: Backend, const C: usize> {
    brightness: Tensor<B, 4>,
    contrast: Tensor<B, 4>,
    saturation: Tensor<B, 4>,
    height: usize,
    width: usize,
    batch_size: usize,
    total_elements: usize,
    ph: PhantomData<B>,
}

impl<B: Backend, const C: usize> ColorJitter<B, C> {
    pub fn new(
        batch_size: usize,
        channels: usize,
        img_width: usize,
        img_height: usize,
        brightness: f32,
        contrast: f32,
        saturation: f32,
        device: &B::Device,
    ) -> ColorJitter<B, C> {
        let mut rng = rand::rng();
        let br_factor = 1.0 + rng.random_range(-brightness..=brightness);
        let ctr_factor = 1.0 + rng.random_range(-contrast..=contrast);
        let sat_factor = 1.0 + rng.random_range(-saturation..=saturation);

        let brightness = Tensor::full(
            Shape::new([batch_size, channels, img_width, img_height]),
            br_factor,
            device,
        );
        let contrast = Tensor::full(
            Shape::new([batch_size, channels, img_width, img_height]),
            ctr_factor,
            device,
        );
        let saturation = Tensor::full(
            Shape::new([batch_size, channels, img_width, img_height]),
            sat_factor,
            device,
        );
        let total_elements = batch_size * channels * img_width * img_height;
        ColorJitter {
            brightness,
            contrast,
            saturation,
            height: img_height,
            width: img_width,
            batch_size,
            total_elements,
            ph: PhantomData,
        }
    }
}

impl<B: Backend, const C: usize> Augmentation<B> for ColorJitter<B, C> {
    fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        // Adjust brightness
        let mut res = (input.clone() * self.brightness.clone()).clamp(0.0, 1.0);

        // Adjust contrast
        let mean = input.clone().mean().into_scalar();
        res = ((res - mean) * self.contrast.clone() + mean).clamp(0.0, 1.0);

        // Adjust saturation
        let r = res
            .clone()
            .slice([0..self.batch_size, 0..1, 0..self.height, 0..self.width]);
        let g = res
            .clone()
            .slice([0..self.batch_size, 1..2, 0..self.height, 0..self.width]);
        let b = res
            .clone()
            .slice([0..self.batch_size, 2..3, 0..self.height, 0..self.width]);

        let gray = r.clone() * 0.2989 + g.clone() * 0.5870 + b.clone() * 0.1140;

        let r_new = ((r - gray.clone()) * self.contrast.clone() + gray.clone()).clamp(0.0, 1.0);
        let g_new = ((g - gray.clone()) * self.contrast.clone() + gray.clone()).clamp(0.0, 1.0);
        let b_new = ((b - gray.clone()) * self.contrast.clone() + gray.clone()).clamp(0.0, 1.0);

        Tensor::cat(vec![r_new, g_new, b_new], 1)
    }
}

#[cfg(test)]
mod tests {}
