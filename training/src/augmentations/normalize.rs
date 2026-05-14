use burn::{
    Tensor,
    tensor::{TensorData, backend::Backend},
};

use crate::augmentations::Augmentation;

#[derive(Clone)]
pub struct Normalize<B: Backend> {
    mean: Tensor<B, 4>,
    std: Tensor<B, 4>,
}

impl<B: Backend> Normalize<B> {
    pub fn new(std: Vec<f32>, mean: Vec<f32>, device: &B::Device) -> Normalize<B> {
        let std_data = TensorData::new(std.clone(), [std.len()]);
        let mean_data = TensorData::new(mean.clone(), [mean.len()]);

        Normalize {
            std: Tensor::<B, 1>::from_data(std_data, device).reshape([1, std.len(), 1, 1]),
            mean: Tensor::<B, 1>::from_data(mean_data, device).reshape([1, mean.len(), 1, 1]),
        }
    }
}

impl<B: Backend> Augmentation<B> for Normalize<B> {
    fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let shape = input.shape();
        let mean: Tensor<B, 4> = self.mean.clone().expand(shape.clone());
        let std: Tensor<B, 4> = self.std.clone().expand(shape);
        input.sub(mean).div(std)
    }
}

#[cfg(test)]
mod tests {}
