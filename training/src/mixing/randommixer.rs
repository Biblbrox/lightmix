use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::Distribution,
};

/// Permuter implementation with indicies
///
/// * `signs`: [TODO:parameter]
/// * `perms`: [TODO:parameter]
/// * `num_heads`: [TODO:parameter]
#[derive(Module, Debug)]
pub struct StaticPermuter<B: Backend> {
    signs: Tensor<B, 3>,
    perms: Tensor<B, 1, Int>,
    num_heads: usize,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub enum PermutationStrategy {
    Random,
    Xor,
}

#[derive(Config, Debug)]
pub struct StaticPermuterConfig {
    embed_dim: usize,
    seq_length: usize,
    num_heads: usize,
    num_encoders: usize,
    strategy: PermutationStrategy,
}

impl<B: Backend> StaticPermuter<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // [H * N, E]
        let perms = self.perms.clone();
        let signs = self.signs.clone();

        // [B, N * H, E]
        let x = x.select(1, perms);

        // [B, N * H, E]
        let x = x * signs;

        self.linear.forward(x.swap_dims(-1, -2)).swap_dims(-1, -2)
    }
}

impl StaticPermuterConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StaticPermuter<B> {
        let distribution = Distribution::Uniform(-1.0, 1.0);
        let d = self.seq_length;
        let mut sign_per_head = Vec::<Tensor<B, 2>>::new();
        let mut perms_per_head = Vec::<Tensor<B, 1, Int>>::new();
        (0..self.num_heads).for_each(|h| {
            let perm: Tensor<B, 1, Int> = match self.strategy {
                PermutationStrategy::Random => {
                    Tensor::<B, 1>::random([d], Distribution::Uniform(0.0, 1.0), device).argsort(0)
                }
                PermutationStrategy::Xor => Tensor::from_data(
                    TensorData::new((0..d).map(|x| (x ^ (h + 1)) as i32).collect(), [d]),
                    device,
                ),
            };

            perms_per_head.push(perm);
            sign_per_head
                .push(Tensor::<B, 2>::random([d, self.embed_dim], distribution, device).sign())
        });
        let perms = Tensor::cat(perms_per_head, 0);
        let signs = Tensor::cat(sign_per_head, 0).unsqueeze();

        StaticPermuter {
            signs,
            perms,
            num_heads: self.num_heads,
            linear: LinearConfig::new(self.seq_length * self.num_heads, self.seq_length)
                .init(device)
                .no_grad(),
        }
    }
}
