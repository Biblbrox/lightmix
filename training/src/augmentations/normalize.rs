use burn::{Tensor, prelude::Backend};

use crate::augmentations::Augmentation;

pub struct Normalize<B: Backend, const C: usize> {
    mean: Tensor<B, 4>,
    std: Tensor<B, 4>,
}

impl<B: Backend, const C: usize> Normalize<B, C> {
    pub fn new(std: [f32; C], mean: [f32; C], device: &B::Device) -> Normalize<B, C> {
        Normalize {
            std: Tensor::<B, 1>::from_floats(std, device).reshape([1, C, 1, 1]),
            mean: Tensor::<B, 1>::from_floats(mean, device).reshape([1, C, 1, 1]),
        }
    }
}

impl<B: Backend, const C: usize> Augmentation<B> for Normalize<B, C> {
    fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let shape = input.shape();
        let mean: Tensor<B, 4> = self.mean.clone().expand(shape.clone());
        let std: Tensor<B, 4> = self.std.clone().expand(shape);
        input.div_scalar(255).sub(mean).div(std)
    }
}

#[cfg(test)]
mod tests {}
