use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::{backend::Backend, ops::PadMode, s},
};

use crate::spectre::transform::SpectralTransform;

pub mod layers;
pub mod transform;

#[derive(Module, Debug)]
pub struct SpectralCompressor<B: Backend> {
    forward_xform: Tensor<B, 2>,
    inverse_xform: Tensor<B, 2>,
}

impl<B: Backend> SpectralCompressor<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, _, image_size] = images.dims();

        // Power-of-two padding
        let xform_size = self.forward_xform.dims()[0];
        let pad_size = xform_size - image_size;
        let x = images.pad((0, pad_size, 0, pad_size), PadMode::Constant(0.0));

        // Forward transform
        let matrix = self.forward_xform.clone().unsqueeze();
        let x = x.matmul(matrix.clone()).t();
        let x = x.matmul(matrix);

        // Low frequency quadrant slice
        let compressed_size = self.inverse_xform.dims()[0];
        let x = x.slice(s![.., .., 0..compressed_size, 0..compressed_size]);

        // Inverse transform
        let matrix = self.inverse_xform.clone().unsqueeze();
        let x = x.matmul(matrix.clone()).t();

        x.matmul(matrix)
    }
}

#[derive(Config, Debug)]
pub struct SpectralCompressConfig {
    spectral: SpectralTransform,
    image_size: usize,
}

impl SpectralCompressConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectralCompressor<B> {
        SpectralCompressor {
            forward_xform: self
                .spectral
                .xform(self.image_size.next_power_of_two().ilog2() as usize, device),
            inverse_xform: self.spectral.xform(
                self.image_size.next_power_of_two().ilog2() as usize - 1,
                device,
            ),
        }
    }
}
