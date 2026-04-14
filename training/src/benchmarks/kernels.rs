use burn::{Tensor, prelude::Backend, tensor::Distribution};
use cubecl::benchmark::Benchmark;

use crate::kernels::hadamard_transform;

pub struct HadamardBench<B: Backend> {
    pub input_shape: Vec<usize>,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for HadamardBench<B> {
    type Input = Tensor<B, 1>;
    type Output = Tensor<B, 1>;

    fn prepare(&self) -> Self::Input {
        Tensor::<B, 1>::random([32], Distribution::Default, &self.device)
    }

    fn name(&self) -> String {
        format!("HadamardBench-{:?}", self.input_shape).to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let custom = hadamard_transform(input);
        Ok(custom)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, Tensor, Tolerance};
    use cubecl::{benchmark::Benchmark, profile::TimingMethod};

    use crate::{
        benchmarks::kernels::HadamardBench,
        kernels::{
            AutodiffBackend, Backend, hadamard_transform, hadamard_transform_reference,
            matmul_add_relu_custom, matmul_add_relu_reference,
        },
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
