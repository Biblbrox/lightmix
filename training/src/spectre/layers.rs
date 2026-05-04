use burn::{
    config::Config,
    module::Module,
    prelude::Backend,
    tensor::{Shape, Tensor, TensorData},
};

use crate::spectre::transform::build_dct_projection;

/// A parameter-free linear layer whose weight matrix is a fixed DCT-II
/// projection.  Drop-in replacement for `Linear<B>` wherever no bias
/// and no learned weights are needed.
///
/// Forward signature matches `Linear`:
///   `Tensor<B, N>` -> `Tensor<B, N>`  (last dim: in_features -> out_features)
#[derive(Module, Debug)]
pub struct DctLinear<B: Backend> {
    weight: Tensor<B, 3>,
}

#[derive(Config, Debug)]
pub struct DctLinearConfig {
    in_features: usize,
    out_features: usize,
}

impl DctLinearConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DctLinear<B> {
        assert!(self.in_features > 0 && self.out_features > 0);

        let data = build_dct_projection(self.in_features, self.out_features);
        let weight = Tensor::<B, 2>::from_data(
            TensorData::new(data, Shape::new([self.out_features, self.in_features])),
            device,
        )
        .unsqueeze_dim(0)
        .transpose();

        DctLinear { weight }
    }
}

impl<B: Backend> DctLinear<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x.matmul(self.weight.clone())
    }
}
