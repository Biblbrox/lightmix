use core::{f32, f64};
use std::marker::PhantomData;

use burn::{Tensor, prelude::Backend};
use burn_vision::Transform2D;
use rand::{Rng, RngExt};

use crate::augmentations::Augmentation;

pub struct RandomAffine<B: Backend> {
    p: f64,
    degrees: f32,
    ph: PhantomData<B>,
}

impl<B: Backend> RandomAffine<B> {
    pub fn new(p: f64, degrees: f32) -> RandomAffine<B> {
        RandomAffine {
            p,
            degrees,
            ph: PhantomData,
        }
    }
}

impl<B: Backend> Augmentation<B> for RandomAffine<B> {
    fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut rng = rand::rng();
        if rng.random_bool(self.p) {
            let rot_mat = Transform2D::rotation(self.degrees.to_radians(), 0.0, 0.0);
            return rot_mat.transform(input);
        };

        input
    }
}

enum Orientation {
    Horizontal,
    Vertical,
}

pub struct RandomFlip<B: Backend> {
    p: f64,
    orientation: Orientation,
    ph: PhantomData<B>,
}

impl<B: Backend> RandomFlip<B> {
    pub fn new(p: f64, orientation: Orientation) -> RandomFlip<B> {
        RandomFlip {
            p,
            orientation,
            ph: PhantomData,
        }
    }
}

impl<B: Backend> Augmentation<B> for RandomFlip<B> {
    fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut rng = rand::rng();
        if rng.random_bool(self.p) {
            match self.orientation {
                Orientation::Vertical => {
                    return input.flip([1]);
                }
                Orientation::Horizontal => {
                    return input.flip([2]);
                }
            }
        }
        input
    }
}

#[cfg(test)]
mod tests {}
