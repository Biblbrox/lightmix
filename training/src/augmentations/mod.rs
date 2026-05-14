use std::marker::PhantomData;

use burn::{prelude::Tensor, tensor::backend::Backend};

pub mod builder;
pub mod colors;
pub mod mix;
pub mod normalize;
pub mod rotation;

pub trait Augmentation<B: Backend>: Send + Sync {
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

    pub fn default() -> Pipeline<B> {
        Pipeline {
            transforms: vec![],
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

    /// Prepends transforms to the front of the pipeline
    pub fn prepend(mut self, mut transforms: Vec<Box<dyn Augmentation<B>>>) -> Self {
        transforms.extend(self.transforms);
        self.transforms = transforms;
        self
    }

    /// Appends transforms to the back of the pipeline
    pub fn append(mut self, transforms: Vec<Box<dyn Augmentation<B>>>) -> Self {
        self.transforms.extend(transforms);
        self
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
        let std = vec![0.5, 0.5, 0.5];
        let mean = vec![0.5, 0.5, 0.5];

        let normalize = Box::new(Normalize::<B>::new(std, mean, &device));
        let random_rotate = Box::new(RandomAffine::<B>::new(0.5, 30.0));
        let color_jitter = Box::new(ColorJitter::<B>::new(0.4, 0.4, 0.4));

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
