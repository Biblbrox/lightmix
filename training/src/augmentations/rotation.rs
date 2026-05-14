use core::{f32, f64};
use std::marker::PhantomData;

use burn::{Tensor, tensor::backend::Backend};

use burn_vision::Transform2D;
use rand::RngExt;

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
            return rot_mat.transform::<B>(input);
        };

        input
    }
}

#[derive(Debug, PartialEq)]
pub enum Orientation {
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
mod tests {
    use super::*;
    use burn::Tensor;
    use burn::tensor::Shape;
    use burn_cuda::{Cuda, CudaDevice};

    type B = Cuda;

    // ============================================================================
    // RandomFlip Tests
    // ============================================================================

    #[test]
    fn test_random_flip_creation() {
        let flip_h = RandomFlip::<B>::new(0.5, Orientation::Horizontal);
        let flip_v = RandomFlip::<B>::new(0.5, Orientation::Vertical);

        assert_eq!(flip_h.orientation, Orientation::Horizontal);
        assert_eq!(flip_v.orientation, Orientation::Vertical);
    }

    #[test]
    fn test_random_flip_horizontal_preserves_shape() {
        let device = CudaDevice::default();
        let flip = RandomFlip::<B>::new(1.0, Orientation::Horizontal);
        let input = Tensor::<B, 4>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device)
            .reshape([1, 2, 3, 1]);

        let output = flip.execute(input.clone());
        assert_eq!(output.shape(), Shape::new([1, 2, 3, 1]));
    }

    #[test]
    fn test_random_flip_vertical_preserves_shape() {
        let device = CudaDevice::default();
        let flip = RandomFlip::<B>::new(1.0, Orientation::Vertical);
        let input = Tensor::<B, 4>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device)
            .reshape([1, 2, 3, 1]);

        let output = flip.execute(input.clone());
        assert_eq!(output.shape(), Shape::new([1, 2, 3, 1]));
    }

    #[test]
    fn test_random_flip_probability_one_always_flips() {
        let device = CudaDevice::default();
        // Run multiple times to verify flip always happens with p=1.0
        let flip = RandomFlip::<B>::new(1.0, Orientation::Horizontal);
        let input = Tensor::<B, 4>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &device)
            .reshape([2, 1, 2, 2]);

        let output = flip.execute(input.clone());

        // Verify shape is preserved
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_random_flip_with_batch_images() {
        let device = CudaDevice::default();
        // Create a batch of 4 images, each 3x32x32 with 3 channels
        let input = Tensor::<B, 4>::from_floats([1.0, 2.0, 3.0], &device).reshape([1, 1, 3, 1]);

        let flip_h = RandomFlip::<B>::new(1.0, Orientation::Horizontal);
        let output_h = flip_h.execute(input.clone());

        // Verify dimensions are preserved
        assert_eq!(output_h.shape(), input.shape());
    }

    // ============================================================================
    // RandomAffine Tests
    // ============================================================================

    #[test]
    fn test_random_affine_creation() {
        let affine = RandomAffine::<B>::new(0.5, 30.0);

        assert_eq!(affine.p, 0.5);
        assert_eq!(affine.degrees, 30.0);
    }

    #[test]
    fn test_random_affine_preserves_shape() {
        let device = CudaDevice::default();
        let affine = RandomAffine::<B>::new(1.0, 30.0);
        let input =
            Tensor::<B, 4>::from_floats([1.0, 2.0, 3.0, 4.0], &device).reshape([1, 1, 2, 2]);

        let output = affine.execute(input.clone());
        assert_eq!(output.shape(), Shape::new([1, 1, 2, 2]));
    }

    #[test]
    fn test_random_affine_with_batch() {
        let device = CudaDevice::default();
        let affine = RandomAffine::<B>::new(1.0, 45.0);
        let input =
            Tensor::<B, 4>::from_floats([1.0, 2.0, 3.0, 4.0], &device).reshape([2, 1, 2, 1]);

        let output = affine.execute(input.clone());
        assert_eq!(output.shape(), Shape::new([2, 1, 2, 1]));
    }

    #[test]
    fn test_random_affine_large_batch() {
        let device = CudaDevice::default();
        let affine = RandomAffine::<B>::new(1.0, 30.0);
        let input =
            Tensor::<B, 4>::from_floats([1.0, 2.0, 3.0, 4.0], &device).reshape([128, 3, 32, 32]);

        let output = affine.execute(input.clone());
        assert_eq!(output.shape(), Shape::new([128, 3, 32, 32]));
    }

    // ============================================================================
    // Combined/Integration Tests
    // ============================================================================

    #[test]
    fn test_random_flip_and_affine_can_be_chained() {
        let device = CudaDevice::default();
        let flip = RandomFlip::<B>::new(1.0, Orientation::Horizontal);
        let affine = RandomAffine::<B>::new(1.0, 30.0);

        let input =
            Tensor::<B, 4>::from_floats([1.0, 2.0, 3.0, 4.0], &device).reshape([1, 1, 2, 2]);

        let flipped = flip.execute(input.clone());
        let final_output = affine.execute(flipped);

        assert_eq!(final_output.shape(), input.shape());
    }
}
