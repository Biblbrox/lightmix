use core::{f32, f64};
use std::marker::PhantomData;

use burn::{Tensor, tensor::backend::Backend};

use burn_vision::Transform2D;
use rand::RngExt;

use crate::augmentations::Augmentation;

#[derive(Debug, Clone)]
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
            let [_, _, h, w] = input.dims();
            let cx = (w as f32 - 1.0) * 0.5;
            let cy = (h as f32 - 1.0) * 0.5;

            let rot_mat = Transform2D::rotation(self.degrees.to_radians(), cx, cy);
            return rot_mat.transform::<B>(input);
        }

        input
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Orientation {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone)]
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
            return match self.orientation {
                Orientation::Vertical => input.flip([2]),
                Orientation::Horizontal => input.flip([3]),
            };
        }
        input
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::Tensor;
    use burn::backend::Flex;
    use burn::backend::flex::FlexDevice;
    use burn::tensor::{Shape, Tolerance};

    type B = Flex;
    type Device = FlexDevice;

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
        let flip = RandomFlip::<B>::new(1.0, Orientation::Horizontal);
        let input = Tensor::<B, 4>::from([[[[1., 2., 3.], [4., 5., 6.]]]]);

        let output = flip.execute(input.clone());
        assert_eq!(output.shape(), Shape::new([1, 1, 2, 3]));
        assert_eq!(input.shape(), Shape::new([1, 1, 2, 3]));
    }

    #[test]
    fn test_random_flip_vertical_preserves_shape() {
        let flip = RandomFlip::<B>::new(1.0, Orientation::Vertical);
        let input = Tensor::<B, 4>::from([[[[1., 2., 3.], [4., 5., 6.]]]]);

        let output = flip.execute(input.clone());
        assert_eq!(output.shape(), Shape::new([1, 1, 2, 3]));
        assert_eq!(input.shape(), Shape::new([1, 1, 2, 3]));
    }

    #[test]
    fn test_random_flip_probability_one_always_preserves_shape() {
        let flip = RandomFlip::<B>::new(1.0, Orientation::Horizontal);
        let input = Tensor::<B, 4>::from([[[[1., 2.], [3., 4.]]]]);

        let output = flip.execute(input.clone());
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_random_flip_with_batch_images() {
        let flip = RandomFlip::<B>::new(1.0, Orientation::Horizontal);
        let input = Tensor::<B, 4>::from([
            [
                [[1., 2.], [3., 4.]],
                [[5., 6.], [7., 8.]],
                [[9., 10.], [11., 12.]],
            ],
            [
                [[13., 14.], [15., 16.]],
                [[17., 18.], [19., 20.]],
                [[21., 22.], [23., 24.]],
            ],
        ]);

        let output = flip.execute(input.clone());
        assert_eq!(output.shape(), input.shape());
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
        let affine = RandomAffine::<B>::new(1.0, 30.0);
        let input = Tensor::<B, 4>::from([[[[1., 2.], [3., 4.]]]]);

        let output = affine.execute(input.clone());
        assert_eq!(output.shape(), Shape::new([1, 1, 2, 2]));
        assert_eq!(input.shape(), Shape::new([1, 1, 2, 2]));
    }

    #[test]
    fn test_random_affine_with_batch() {
        let affine = RandomAffine::<B>::new(1.0, 45.0);
        let input = Tensor::<B, 4>::from([[[[1., 2.], [3., 4.]]], [[[5., 6.], [7., 8.]]]]);

        let output = affine.execute(input.clone());
        assert_eq!(output.shape(), Shape::new([2, 1, 2, 2]));
    }

    #[test]
    fn test_random_affine_large_batch() {
        let device = Device::default();

        let affine = RandomAffine::<B>::new(1.0, 30.0);
        let input = Tensor::<B, 4>::zeros([128, 3, 32, 32], &device);

        let output = affine.execute(input.clone());
        assert_eq!(output.shape(), Shape::new([128, 3, 32, 32]));
    }

    // ============================================================================
    // Combined/Integration Tests
    // ============================================================================

    #[test]
    fn test_random_flip_and_affine_can_be_chained() {
        let flip = RandomFlip::<B>::new(1.0, Orientation::Horizontal);
        let affine = RandomAffine::<B>::new(1.0, 30.0);

        let input = Tensor::<B, 4>::from([[[[1., 2.], [3., 4.]]]]);

        let flipped = flip.execute(input.clone());
        let final_output = affine.execute(flipped);

        assert_eq!(final_output.shape(), input.shape());
    }

    #[test]
    fn test_random_flip_horizontal_values() {
        let flip = RandomFlip::<B>::new(1.0, Orientation::Horizontal);

        let input = Tensor::<B, 4>::from([[[[1., 2., 3.], [4., 5., 6.]]]]);
        let output = flip.execute(input);

        let expected = Tensor::<B, 4>::from([[[[3., 2., 1.], [6., 5., 4.]]]]);

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_random_flip_vertical_values() {
        let flip = RandomFlip::<B>::new(1.0, Orientation::Vertical);

        let input = Tensor::<B, 4>::from([[[[1., 2., 3.], [4., 5., 6.]]]]);
        let output = flip.execute(input);

        let expected = Tensor::<B, 4>::from([[[[4., 5., 6.], [1., 2., 3.]]]]);

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_random_affine_rotation_45_values() {
        let affine = RandomAffine::<B>::new(1.0, 45.0);

        let input = Tensor::<B, 4>::from([[[[1., 2.], [3., 4.]]]]);

        let output = affine.execute(input.clone());

        let expected = Tensor::<B, 4>::from([[[[1.7500, 2.7929], [1.8358, 3.7500]]]]);

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_random_affine_rotation_90_values() {
        let affine = RandomAffine::<B>::new(1.0, 90.0);

        let input = Tensor::<B, 4>::from([[[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]]);

        let output = affine.execute(input.clone());

        let expected = Tensor::<B, 4>::from([[[[3., 6., 9.], [3., 6., 9.], [3., 6., 9.]]]]);

        expected
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn test_random_flip_probability_zero_returns_unchanged() {
        let flip = RandomFlip::<B>::new(0.0, Orientation::Horizontal);
        let input = Tensor::<B, 4>::from([[[[1., 2., 3.], [4., 5., 6.]]]]);

        let output = flip.execute(input.clone());

        // With p=0.0, should never flip - output equals input
        input
            .to_data()
            .assert_approx_eq(&output.to_data(), Tolerance::<f32>::balanced());
    }
}
