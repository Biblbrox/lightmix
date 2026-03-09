use burn::{Tensor, prelude::Backend};

use crate::augmentations::Augmentation;

pub struct Normalize<B: Backend, const C: usize> {
    mean: Tensor<B, 4>,
    std: Tensor<B, 4>,
}

impl<B: Backend, const C: usize> Normalize<B, C> {
    pub fn new(
        batch_size: usize,
        channels: usize,
        img_width: usize,
        img_height: usize,
        std: [f32; C],
        mean: [f32; C],
        device: &B::Device,
    ) -> Normalize<B, C> {
        Normalize {
            mean: Tensor::<B, 1>::from_floats(mean, device)
                .reshape([1, channels, 1, 1])
                .expand([batch_size, channels, img_width, img_height]),

            std: Tensor::<B, 1>::from_floats(std, device)
                .reshape([1, channels, 1, 1])
                .expand([batch_size, 3, img_width, img_height]),
        }
    }
}

impl<B: Backend, const C: usize> Augmentation<B> for Normalize<B, C> {
    fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        input
            .swap_dims(1, -1)
            .div_scalar(255)
            .sub(self.mean.clone())
            .div(self.std.clone())
    }
}

#[cfg(test)]
mod tests {}
