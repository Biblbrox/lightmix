use std::marker::PhantomData;

use burn::prelude::*;

pub mod colors;
pub mod normalize;
pub mod rotation;

pub trait Augmentation<B: Backend> {
    fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4>;
}

pub struct Pipeline<B: Backend> {
    transforms: Vec<Box<dyn Augmentation<B>>>,
    ph: PhantomData<B>,
}

impl<B: Backend> Pipeline<B> {
    pub fn new(transforms: Vec<Box<dyn Augmentation<B>>>) -> Pipeline<B> {
        Pipeline {
            transforms,
            ph: PhantomData,
        }
    }

    pub fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.transforms.is_empty() {
            return input;
        }

        let mut res = self.transforms[0].execute(input);
        if self.transforms.len() == 1 {
            return res;
        }
        for tr in self.transforms.iter().skip(1) {
            res = tr.execute(res);
        }

        res
    }
}

#[cfg(test)]
mod tests {
    use burn::{Tensor, tensor::Shape};
    use burn_cuda::{Cuda, CudaDevice};

    use crate::augmentations::{
        Augmentation, Pipeline, colors::ColorJitter, normalize::Normalize, rotation::RandomAffine,
    };

    #[test]
    fn test_pipeline() {
        type B = Cuda;
        let device = CudaDevice::default();
        let std = [0.5, 0.5, 0.5];
        let mean = [0.5, 0.5, 0.5];

        let normalize = Box::new(Normalize::<B, 3>::new(128, 3, 32, 32, std, mean, &device));
        let random_rotate = Box::new(RandomAffine::<B>::new(0.5, 30.0));
        let color_jitter = Box::new(ColorJitter::<B, 3>::new(
            128, 3, 32, 32, 0.4, 0.4, 0.4, &device,
        ));

        let transforms: Vec<Box<dyn Augmentation<B>>> =
            vec![normalize, random_rotate, color_jitter];
        let pipeline = Pipeline::new(transforms);
        let input = Tensor::<B, 4>::random(
            Shape::new([128, 32, 32, 3]),
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let res = pipeline.execute(input);
    }
}
