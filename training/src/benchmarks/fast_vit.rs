use burn::{Tensor, prelude::Backend, tensor::Distribution};
use cubecl::benchmark::Benchmark;

use crate::models::fast_vit::{FastViT, FastViTConfig};

pub struct FastViTBenchmark<B: Backend> {
    pub num_patches: usize,
    pub batch_size: usize,
    pub in_channels: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub num_classes: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub hid_dim: usize,
    pub dropout: f64,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for FastViTBenchmark<B> {
    type Input = (Tensor<B, 4>, FastViT<B>);
    type Output = Tensor<B, 2>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 4>::random(
                [
                    self.batch_size,
                    self.in_channels,
                    self.image_size,
                    self.image_size,
                ],
                Distribution::Default,
                &self.device,
            ),
            FastViTConfig::new(
                self.in_channels,
                self.embed_dim,
                self.num_heads,
                self.num_layers,
                self.num_classes,
                self.patch_size,
                self.image_size,
                self.hid_dim,
                self.dropout,
                0.05,
            )
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "FastViT-{:?}x{:?}x{:?} {:?} heads",
            self.batch_size, self.num_patches, self.embed_dim, self.num_heads
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, model) = input;
        let res = model.forward(tensor);
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use cubecl::{
        benchmark::{Benchmark, BenchmarkComputations},
        profile::TimingMethod,
    };

    use crate::{
        benchmarks::{GpuBackend, GpuDevice, fast_vit::FastViTBenchmark},
        utils::print_bench_results,
    };

    #[test]
    fn fast_vit_bench() {
        let device = GpuDevice::default();

        let batches = [8; 5];
        let embed_dim = [64, 128, 256, 512, 1024];
        let num_tokens = [65; 5];
        let num_heads = [1, 2, 3, 4, 5, 6, 7, 8];
        let out_channels = [64, 128, 256, 512, 1024];
        let image_size: usize = 224;
        let patch_size: usize = 16;
        let num_patches: usize = (image_size / patch_size).pow(2);
        let num_classes = 1000;
        let hid_dim = 768;
        let in_channels = 3;
        let dropout = 0.5;

        let mut results: Vec<(u32, BenchmarkComputations)> = Vec::new();
        for head in num_heads.into_iter() {
            let bench = FastViTBenchmark::<GpuBackend> {
                num_patches,
                batch_size: batches[0],
                embed_dim: embed_dim[0],
                num_heads: head,
                num_layers: num_tokens[0],
                in_channels,
                num_classes,
                patch_size,
                image_size,
                hid_dim,
                device: device.clone(),
                dropout,
            };

            let bench_res = bench.run(TimingMethod::System).unwrap();
            let computed = BenchmarkComputations::new(&bench_res);
            results.push((head as u32, computed));
        }

        print_bench_results("FastViT", &results, "num_heads");
    }
}
