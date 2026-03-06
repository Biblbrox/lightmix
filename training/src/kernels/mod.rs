pub mod backward;
pub mod benchmark;
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

#[cfg(test)]
mod tests {
    use burn::{
        backend::wgpu::WgpuRuntime,
        tensor::{Distribution, Tensor, Tolerance},
    };
    use cubecl::{benchmark::Benchmark, profile::TimingMethod};

    use crate::kernels::{
        AutodiffBackend, Backend, benchmark::HadamardBench, hadamard_transform,
        hadamard_transform_reference, matmul_add_relu_custom, matmul_add_relu_reference,
    };

    fn inference_matmut_add_relu<B: Backend>(device: &B::Device) {
        let lhs = Tensor::<B, 3>::random([1, 32, 32], Distribution::Default, device);
        let rhs = Tensor::random([32, 32, 32], Distribution::Default, device);
        let bias = Tensor::random([32, 32, 32], Distribution::Default, device);

        let reference = matmul_add_relu_reference(lhs.clone(), rhs.clone(), bias.clone())
            .into_data()
            .convert::<f32>();
        let custom = matmul_add_relu_custom(lhs, rhs, bias)
            .into_data()
            .convert::<f32>();

        reference.assert_approx_eq::<f32>(&custom, Tolerance::default());

        println!("Both reference and the custom fused kernel have the same output");
    }

    fn inference_hadamard_transform<B: Backend>(device: &B::Device) {
        let mut data = Tensor::<B, 1>::random([32], Distribution::Default, device);
        let data_kernel = data.clone();
        let reference_result = hadamard_transform_reference(&mut data);
        let custom = hadamard_transform(data_kernel.clone())
            .into_data()
            .convert::<f32>();

        custom.assert_approx_eq::<f32>(
            &reference_result.into_data().convert::<f32>(),
            Tolerance::default(),
        );

        println!("Both hadamard reference and its custom kernel have the same output");
    }
    fn autodiff_matmul_add_relu<B: AutodiffBackend>(device: &B::Device) {
        let lhs = Tensor::<B, 3>::random([1, 32, 32], Distribution::Default, device).require_grad();
        let rhs = Tensor::random([32, 32, 32], Distribution::Default, device).require_grad();
        let bias = Tensor::random([32, 32, 32], Distribution::Default, device).require_grad();

        let reference = matmul_add_relu_reference(lhs.clone(), rhs.clone(), bias.clone());

        let mut gradients = reference.backward();

        let lhs_grad_ref = lhs.grad_remove(&mut gradients).unwrap();
        let rhs_grad_ref = rhs.grad_remove(&mut gradients).unwrap();
        let bias_grad_ref = bias.grad_remove(&mut gradients).unwrap();

        let lhs = lhs.detach();
        let rhs = rhs.detach();
        let bias = bias.detach();

        let custom = matmul_add_relu_custom(lhs.clone(), rhs.clone(), bias.clone());

        let mut gradients = custom.backward();

        let lhs_grad_custom = lhs.grad_remove(&mut gradients).unwrap();
        let rhs_grad_custom = rhs.grad_remove(&mut gradients).unwrap();
        let bias_grad_custom = bias.grad_remove(&mut gradients).unwrap();

        lhs_grad_ref
            .into_data()
            .convert::<B::FloatElem>()
            .assert_approx_eq::<f32>(
                &lhs_grad_custom.into_data().convert::<B::FloatElem>(),
                Tolerance::default(),
            );

        println!("Both reference and the custom fused kernel have the same lhs gradient");

        rhs_grad_ref
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &rhs_grad_custom.into_data().convert::<B::FloatElem>(),
                Tolerance::default(),
            );

        println!("Both reference and the custom fused kernel have the same rhs gradient");

        bias_grad_ref
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &bias_grad_custom.into_data().convert::<B::FloatElem>(),
                Tolerance::default(),
            );

        println!("Both reference and the custom fused kernel have the same bias gradient");
    }

    #[test]
    fn test_custom_kernel() {
        type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
        type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;
        let device = Default::default();
        inference_matmut_add_relu::<MyBackend>(&device);
        inference_hadamard_transform::<MyBackend>(&device);
        autodiff_matmul_add_relu::<MyAutodiffBackend>(&device);
    }

    pub fn launch<B: Backend>(device: B::Device) {
        let bench1 = HadamardBench::<B> {
            input_shape: vec![512, 8 * 1024],
            device: device.clone(),
        };
        let bench2 = HadamardBench::<B> {
            input_shape: vec![512, 8 * 1024],
            device,
        };

        for bench in [bench1, bench2] {
            println!("{}", bench.name());
            println!("{}", bench.run(TimingMethod::System).unwrap());
        }
    }

    #[test]
    fn test_benchmark_hadamard() {
        type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
        type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;
        launch::<MyBackend>(Default::default());
    }
}
