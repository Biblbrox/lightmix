use burn::{
    module::{Module, Param, Parameter},
    nn::{
        Linear, LinearConfig,
        conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::Distribution,
};
use burn_cubecl::kernel::matmul::matmul;
use rand::{Rng, RngExt, SeedableRng, rngs::{self, SmallRng}, seq::SliceRandom};

/// Permuter implementation with indicies
///
/// * `signs`: [TODO:parameter]
/// * `perms`: [TODO:parameter]
/// * `num_heads`: [TODO:parameter]
#[derive(Module, Debug)]
pub struct Permuter<B: Backend> {
    signs: Tensor<B, 2>,
    perms: Tensor<B, 1, Int>,
    num_heads: usize,
}

/// Permuter implementation with permutation matrix
///
/// * `signs`: [TODO:parameter]
/// * `perms`: [TODO:parameter]
/// * `num_heads`: [TODO:parameter]
#[derive(Module, Debug)]
pub struct PermuterMatrix<B: Backend> {
    signs: Tensor<B, 3>,
    perms: Tensor<B, 1, Int>,
    params: Param<Tensor<B, 3>>,
    num_heads: usize,
    embed_dim: usize,
    seq_length: usize,
}

#[derive(Config, Debug)]
pub struct PermuterConfig {
    embed_dim: usize,
    seq_length: usize,
    num_heads: usize,
    out_channels: usize,
    num_encoders: usize,
}

#[derive(Config, Debug)]
pub struct PermuterMatrixConfig {
    embed_dim: usize,
    seq_length: usize,
    num_heads: usize,
    out_channels: usize,
    num_encoders: usize,
}

impl<B: Backend> Permuter<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let shape = x.shape();

        // [B, N * E]
        let x = x.reshape([shape[0], shape[1] * shape[2]]);
        debug_assert_eq!(x.shape(), Shape::new([shape[0], shape[1] * shape[2]]));

        // [H * N * E]
        let perms = self.perms.clone();
        let signs = self.signs.clone();

        // [B, N * H * E]
        let x = x.select(1, perms);

        // [B, N * H * E]
        let x = x * signs;
        x.reshape([shape[0], shape[1], shape[2] * self.num_heads])
    }
}

impl<B: Backend> PermuterMatrix<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // [1, H * N, 1]
        let perms = self.perms.clone();

        // [B, H * N, E]
        let signs = self.signs.clone();
        // [B, H * N, E]
        let x = x.select(1, perms) * signs;

        let params = self.params.val().repeat(&[1, 4, 4]);
        (x * params).swap_dims(2, 1)
    }
}

impl PermuterMatrixConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PermuterMatrix<B> {
        let distribution = Distribution::Uniform(-1.0, 1.0);
        let d = self.seq_length;
        let mut sign_per_head = Vec::<Tensor<B, 3>>::new();
        let mut perms_per_head = Vec::<Tensor<B, 1, Int>>::new();

        (0..self.num_heads).for_each(|h| {
            let rand_indices =
                Tensor::<B, 1>::random(Shape::new([d]), Distribution::Uniform(0.0, 1.0), device)
                    .argsort(0);

            let signs = Tensor::<B, 2>::random([self.seq_length, self.embed_dim], distribution, device).sign();
            perms_per_head.push(rand_indices);
            sign_per_head.push(signs.unsqueeze_dim::<3>(0));
        });
        let perms = Tensor::cat(perms_per_head, 0);
        let signs = Tensor::cat(sign_per_head, 1);

        PermuterMatrix {
            signs,
            perms,
            num_heads: self.num_heads,
            seq_length: self.seq_length,
            embed_dim: self.embed_dim,
            params: Param::from_tensor(Tensor::<B, 2>::ones([self.seq_length * self.num_heads / 4, self.embed_dim / 4], device).unsqueeze_dim::<3>(0)).set_require_grad(true)
        }
    }
}


impl PermuterConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Permuter<B> {
        let distribution = Distribution::Uniform(-1.0, 1.0);
        let d = self.embed_dim * self.seq_length;
        let mut sign_per_head = Vec::<Tensor<B, 1>>::new();
        let mut perms_per_head = Vec::<Tensor<B, 1, Int>>::new();
        (0..self.num_heads).for_each(|_| {
            let rand_indices =
                Tensor::<B, 1>::random([d], Distribution::Uniform(0.0, 1.0), device).argsort(0);

            perms_per_head.push(rand_indices);
            sign_per_head.push(Tensor::<B, 1>::random([d], distribution, device).sign())
        });
        let perms = Tensor::cat(perms_per_head, 0);
        let signs = Tensor::cat(sign_per_head, 0).unsqueeze();

        Permuter {
            signs,
            perms,
            num_heads: self.num_heads,
        }
    }
}

#[derive(Module, Debug)]
pub struct MHPermutMix<B: Backend> {
    permuter: Permuter<B>,
    linear: Linear<B>,
    num_heads: usize,
}

#[derive(Config, Debug)]
pub struct MHPermutMixConfig {
    embed_dim: usize,
    seq_length: usize,
    num_heads: usize,
    out_channels: usize,
    num_encoders: usize,
}

impl<B: Backend> MHPermutMix<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let out = self.permuter.forward(x);
        self.linear.forward(out)
    }
}

impl MHPermutMixConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MHPermutMix<B> {
        MHPermutMix {
            permuter: PermuterConfig::new(
                self.embed_dim,
                self.seq_length,
                self.num_heads,
                self.out_channels,
                self.num_encoders,
            )
            .init(device),
            linear: LinearConfig::new(self.embed_dim * self.num_heads, self.out_channels)
                .init(device),
            num_heads: self.num_heads,
        }
    }
}

#[derive(Module, Debug)]
pub struct MHPermutMixMatrix<B: Backend> {
    permuter: PermuterMatrix<B>,
    linear: Linear<B>,
    num_heads: usize,
}

#[derive(Config, Debug)]
pub struct MHPermutMixMatrixConfig {
    embed_dim: usize,
    seq_length: usize,
    num_heads: usize,
    out_channels: usize,
    num_encoders: usize,
    encoder: usize
}

impl<B: Backend> MHPermutMixMatrix<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let out = self.permuter.forward(x);
        self.linear.forward(out).swap_dims(1, 2)
    }
}

impl MHPermutMixMatrixConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MHPermutMixMatrix<B> {
        MHPermutMixMatrix {
            permuter: PermuterMatrixConfig::new(
                self.embed_dim,
                self.seq_length,
                self.num_heads,
                self.out_channels,
                self.num_encoders,
            )
            .init(device),
            linear: LinearConfig::new(self.seq_length * self.num_heads, self.seq_length).with_bias(false)
                .init(device),
            num_heads: self.num_heads,
        }
    }
}
