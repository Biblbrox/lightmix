use std::marker::PhantomData;

use burn::tensor::ops::FloatTensor;
use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::client::ComputeClient;
use cubecl::{future, prelude::*};

//pub struct HadamardBench<R: Runtime, F: Float + CubeElement> {
//    input_shape: Vec<usize>,
//    client: ComputeClient<R::Server>,
//    _f: PhantomData<F>,
//}
//
//impl<R: Runtime, F: Float + CubeElement> Benchmark for HadamardBench<R, F> {
//    type Input = FloatTensor<F>;
//    type Output = FloatTensor<F>;
//
//    fn prepare(&self) -> Self::Input {
//        GpuTensor::<R, F>::arange(self.input_shape.clone(), &self.client)
//    }
//
//    fn name(&self) -> String {
//        format!("{}-reduction-{:?}", R::name(&self.client), self.input_shape).to_lowercase()
//    }
//
//    fn sync(&self) {
//        future::block_on(self.client.sync())
//    }
//
//    fn execute(&self, input: Self::Input) -> Self::Output {
//        let output_shape: Vec<usize> = vec![self.input_shape[0]];
//        let output = GpuTensor::<R, F>::empty(output_shape, &self.client);
//
//        unsafe {
//            reduce_matrix::launch_unchecked::<F, R>(
//                &self.client,
//                CubeCount::Static(1, 1, 1),
//                CubeDim::new(1, 1, 1),
//                input.into_tensor_arg(1),
//                output.into_tensor_arg(1),
//            );
//        }
//
//        output
//    }
//}
