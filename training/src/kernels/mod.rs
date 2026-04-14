pub mod backward;
pub mod forward;
pub mod hadamard;
pub mod kernel;

use burn::tensor::{Tensor, TensorData, TensorPrimitive, activation, ops::FloatTensor};

/// We create our own Backend trait that extends the Burn backend trait.
pub trait Backend: burn::tensor::backend::Backend {
    fn fused_matmul_add_relu(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    fn hadamard_transform_inner(data: FloatTensor<Self>) -> FloatTensor<Self>;
}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}

/// We define our custom implementation using the added function on our custom backend.
pub fn matmul_add_relu_custom<B: Backend>(
    lhs: Tensor<B, 3>,
    rhs: Tensor<B, 3>,
    bias: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let output = B::fused_matmul_add_relu(
        lhs.into_primitive().tensor(),
        rhs.into_primitive().tensor(),
        bias.into_primitive().tensor(),
    );

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

pub fn hadamard_transform<B: Backend>(data: Tensor<B, 1>) -> Tensor<B, 1> {
    let output = B::hadamard_transform_inner(data.into_primitive().tensor());

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// We define a reference implementation using basic tensor operations.
pub fn matmul_add_relu_reference<B: Backend>(
    lhs: Tensor<B, 3>,
    rhs: Tensor<B, 3>,
    bias: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let x = lhs.matmul(rhs) + bias;

    activation::relu(x)
}

/// Reference hadamard transform implementation from https://github.com/edugzlez/fwht/blob/master/src/core.rs
pub fn hadamard_transform_reference<B: Backend>(data: &mut Tensor<B, 1>) -> Tensor<B, 1> {
    let n = data.shape().dims[0];

    let mut h = 1;
    let mut out: Vec<f32> = vec![0.0; n];
    let data_native = data.clone().into_data();
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data_native.clone().as_slice::<f32>().unwrap()[j];
                let y = data_native.clone().as_slice::<f32>().unwrap()[j + h];
                out[j] = x + y;
                out[j + h] = x - y;
            }
        }
        h *= 2;
    }

    Tensor::<B, 1>::from_data(TensorData::new(out, [n]), &data.device())
}
