use burn::{Tensor, prelude::Backend, tensor::Distribution};
use cubecl::{benchmark::Benchmark, future};

use crate::spectre_vit::{MHPermutMix, MHPermutMixConfig};

pub struct MHPermutMixBenchmark<B: Backend> {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub out_channels: usize,
    pub device: B::Device,
}

impl<B: Backend> Benchmark for MHPermutMixBenchmark<B> {
    type Input = (Tensor<B, 3>, MHPermutMix<B>);
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        (
            Tensor::<B, 3>::random(
                [self.batch_size, self.num_tokens, self.embed_dim],
                Distribution::Default,
                &self.device,
            ),
            MHPermutMixConfig::new(
                self.embed_dim,
                self.num_tokens,
                self.num_heads,
                self.out_channels,
                1,
            )
            .init(&self.device),
        )
    }

    fn name(&self) -> String {
        format!(
            "MHPermutMix-{:?}x{:?}x{:?} {:?} heads",
            self.batch_size, self.num_tokens, self.embed_dim, self.num_heads
        )
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (tensor, mh_permut) = input;
        let res = mh_permut.forward(tensor, 0);
        Ok(res)
    }
}
