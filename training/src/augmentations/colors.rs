use std::marker::PhantomData;

use burn::{Tensor, prelude::Backend, tensor::Shape};
use rand::RngExt;

use crate::augmentations::Augmentation;

pub struct ColorJitter<B: Backend, const C: usize> {
    brightness: Tensor<B, 4>,
    contrast: Tensor<B, 4>,
    saturation: Tensor<B, 4>,
    ph: PhantomData<B>,
}

impl<B: Backend, const C: usize> ColorJitter<B, C> {
    pub fn new(
        brightness: f32,
        contrast: f32,
        saturation: f32,
        device: &B::Device,
    ) -> ColorJitter<B, C> {
        let mut rng = rand::rng();
        let br = [1.0 + rng.random_range(-brightness..=brightness); C];
        let ctr = [1.0 + rng.random_range(-contrast..=contrast); C];
        let st = [1.0 + rng.random_range(-saturation..=saturation); C];

        let brightness = Tensor::<B, 1>::from_floats(br, device).reshape([1, C, 1, 1]);
        let contrast = Tensor::<B, 1>::from_floats(ctr, device).reshape([1, C, 1, 1]);
        let saturation = Tensor::<B, 1>::from_floats(st, device).reshape([1, C, 1, 1]);

        ColorJitter {
            brightness,
            contrast,
            saturation,
            ph: PhantomData,
        }
    }
}

impl<B: Backend, const C: usize> Augmentation<B> for ColorJitter<B, C> {
    fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let shape = input.shape();
        let brightness = self.brightness.clone().expand(shape.clone());
        let contrast = self.contrast.clone().expand(shape.clone());
        let saturation = self.saturation.clone().expand(shape.clone());

        // Adjust brightness
        let mut res = (input.clone() * brightness.clone()).clamp(0.0, 1.0);

        // Adjust contrast
        let mean = input.clone().mean().into_scalar();
        res = ((res - mean) * contrast.clone() + mean).clamp(0.0, 1.0);

        // Adjust saturation
        let r = res
            .clone()
            .slice([0..shape[0], 0..1, 0..shape[2], 0..shape[3]]);
        let g = res
            .clone()
            .slice([0..shape[0], 1..2, 0..shape[2], 0..shape[3]]);
        let b = res
            .clone()
            .slice([0..shape[0], 2..3, 0..shape[2], 0..shape[3]]);

        let gray = r.clone() * 0.2989 + g.clone() * 0.5870 + b.clone() * 0.1140;

        let r_new = ((r - gray.clone()) * saturation.clone() + gray.clone()).clamp(0.0, 1.0);
        let g_new = ((g - gray.clone()) * saturation.clone() + gray.clone()).clamp(0.0, 1.0);
        let b_new = ((b - gray.clone()) * saturation.clone() + gray.clone()).clamp(0.0, 1.0);

        Tensor::cat(vec![r_new, g_new, b_new], 1)
    }
}

#[cfg(test)]
mod tests {}
