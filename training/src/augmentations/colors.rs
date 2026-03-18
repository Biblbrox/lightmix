use std::marker::PhantomData;

use burn::{Tensor, prelude::Backend, tensor::Shape};
use rand::RngExt;

use crate::augmentations::Augmentation;

pub struct ColorJitter<B: Backend> {
    brightness: f32,
    contrast: f32,
    saturation: f32,
    ph: PhantomData<B>,
}

pub struct RandomGrayscale<B: Backend> {
    p: f64,
    ph: PhantomData<B>,
}

impl<B: Backend> ColorJitter<B> {
    pub fn new(
        brightness: f32,
        contrast: f32,
        saturation: f32,
        device: &B::Device,
    ) -> ColorJitter<B> {
        ColorJitter {
            brightness,
            contrast,
            saturation,
            ph: PhantomData,
        }
    }
}

impl<B: Backend> RandomGrayscale<B> {
    pub fn new(p: f64, device: &B::Device) -> RandomGrayscale<B> {
        RandomGrayscale { p, ph: PhantomData }
    }
}

impl<B: Backend> Augmentation<B> for ColorJitter<B> {
    // Input shape: [B, C, H, W]
    fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let shape = input.shape();

        let mut rng = rand::rng();
        let br = [1.0 + rng.random_range(-self.brightness..=self.brightness); 1];
        let ctr = [1.0 + rng.random_range(-self.contrast..=self.contrast); 1];
        let st = [1.0 + rng.random_range(-self.saturation..=self.saturation); 1];

        let brightness = Tensor::<B, 1>::from_floats(br, &input.device()).reshape([1, 1, 1, 1]);
        let contrast = Tensor::<B, 1>::from_floats(ctr, &input.device()).reshape([1, 1, 1, 1]);
        let saturation = Tensor::<B, 1>::from_floats(st, &input.device()).reshape([1, 1, 1, 1]);

        let brightness = brightness
            .clone()
            .expand(Shape::new([shape[0], 1, shape[2], shape[3]]));
        let contrast = contrast
            .clone()
            .expand(Shape::new([shape[0], 1, shape[2], shape[3]]));
        let saturation = saturation
            .clone()
            .expand(Shape::new([shape[0], 1, shape[2], shape[3]]));

        // Adjust brightness
        let mut res = (input.clone() * brightness).clamp(0.0, 1.0);

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

impl<B: Backend> Augmentation<B> for RandomGrayscale<B> {
    // Input shape: [B, C, H, W]
    fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let shape = input.shape();

        let mut rng = rand::rng();
        if rng.random_bool(self.p) {
            let r = input
                .clone()
                .slice([0..shape[0], 0..1, 0..shape[2], 0..shape[3]]);
            let g = input
                .clone()
                .slice([0..shape[0], 1..2, 0..shape[2], 0..shape[3]]);
            let b = input
                .clone()
                .slice([0..shape[0], 2..3, 0..shape[2], 0..shape[3]]);

            let gray = r.clone() * 0.2989 + g.clone() * 0.5870 + b.clone() * 0.1140;
            return Tensor::cat(vec![gray.clone(), gray.clone(), gray], 1);
        }
        return input;
    }
}

#[cfg(test)]
mod tests {}
