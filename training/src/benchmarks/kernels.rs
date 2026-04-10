use burn::{Tensor, tensor::Distribution};
use cubecl::{benchmark::Benchmark, future};

use crate::kernels::{Backend, hadamard_transform};

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
