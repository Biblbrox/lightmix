use burn::{
    module::{Module, Parameter},
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::Distribution,
};
use rand::{rngs, seq::SliceRandom};

//#[derive(Module, Debug)]
//pub struct Permuter<B: Backend> {
//    signs: Vec<Tensor<B, 2>>,
//    perms: Vec<Tensor<B, 1, Int>>,
//    num_heads: usize,
//}
//
//#[derive(Config, Debug)]
//pub struct PermuterConfig {
//    embed_dim: usize,
//    seq_length: usize,
//    num_heads: usize,
//    out_channels: usize,
//    num_encoders: usize,
//}
//
//impl<B: Backend> Permuter<B> {
//    pub fn forward(&self, x: Tensor<B, 3>, encoder_num: usize) -> Tensor<B, 3> {
//        let shape = x.shape();
//
//        // [B, N * E]
//        let x = x.reshape([shape[0], shape[1] * shape[2]]);
//        debug_assert_eq!(x.shape(), Shape::new([shape[0], shape[1] * shape[2]]));
//
//        // [H * N * E]
//        let perms = self.perms[encoder_num].clone();
//        let signs = self.signs[encoder_num].clone();
//
//        // [B, N * H * E]
//        let x = x.select(1, perms);
//
//        // [B, N * H * E]
//        let x = x * signs;
//        x.reshape([shape[0], shape[1], shape[2] * self.num_heads])
//    }
//}
//
//impl PermuterConfig {
//    pub fn init<B: Backend>(&self, device: &B::Device) -> Permuter<B> {
//        let distribution = Distribution::Uniform(-1.0, 1.0);
//        let d = self.embed_dim * self.seq_length;
//        let mut perms_per_encoder = Vec::<Tensor<B, 1, Int>>::new();
//        let mut sign_per_encoder = Vec::<Tensor<B, 2>>::new();
//        (0..self.num_encoders).for_each(|_| {
//            let mut sign_per_head = Vec::<Tensor<B, 1>>::new();
//            let mut perms_per_head = Vec::<Tensor<B, 1, Int>>::new();
//            (0..self.num_heads).for_each(|_| {
//                let rand_indices = Tensor::<B, 1>::random(
//                    Shape::new([d]),
//                    Distribution::Uniform(0.0, 1.0),
//                    device,
//                )
//                .argsort(0)
//                .set_require_grad(false);
//
//                perms_per_head.push(rand_indices);
//                sign_per_head.push(
//                    Tensor::<B, 1>::random([d], distribution, device)
//                        .sign()
//                        .set_require_grad(false),
//                )
//            });
//            perms_per_encoder.push(Tensor::cat(perms_per_head, 0));
//            sign_per_encoder.push(Tensor::cat(sign_per_head, 0).unsqueeze());
//        });
//
//        Permuter {
//            signs: sign_per_encoder,
//            perms: perms_per_encoder,
//            num_heads: self.num_heads,
//        }
//        .no_grad()
//    }
//}
//

#[derive(Module, Debug)]
pub struct Permuter<B: Backend> {
    signs: Vec<Tensor<B, 3>>,
    perms: Vec<Tensor<B, 1, Int>>,
    num_heads: usize,
}

#[derive(Config, Debug)]
pub struct PermuterConfig {
    embed_dim: usize,
    seq_length: usize,
    num_heads: usize,
    out_channels: usize,
    num_encoders: usize,
}

impl<B: Backend> Permuter<B> {
    pub fn forward(&self, x: Tensor<B, 3>, encoder_num: usize) -> Tensor<B, 3> {
        let shape = x.shape();

        // [B, N * E]
        //let x = x.reshape([shape[0], shape[1] * shape[2]]);
        //debug_assert_eq!(x.shape(), Shape::new([shape[0], shape[1] * shape[2]]));

        // [H * N]
        let perms = self.perms[encoder_num].clone();
        let signs = self.signs[encoder_num].clone();

        // [B, N * H, E]
        let x = x.select(1, perms);

        // [B, N * H, E]
        let x = x * signs;

        // [B, N, E * H]
        x.reshape([shape[0], shape[1], shape[2] * self.num_heads])
    }
}

impl PermuterConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Permuter<B> {
        let distribution = Distribution::Uniform(-1.0, 1.0);
        let d = self.seq_length;
        let mut perms_per_encoder = Vec::<Tensor<B, 1, Int>>::new();
        let mut sign_per_encoder = Vec::<Tensor<B, 3>>::new();
        (0..self.num_encoders).for_each(|_| {
            let mut sign_per_head = Vec::<Tensor<B, 3>>::new();
            let mut perms_per_head = Vec::<Tensor<B, 1, Int>>::new();
            (0..self.num_heads).for_each(|_| {
                let rand_indices = Tensor::<B, 1>::random(
                    Shape::new([d]),
                    Distribution::Uniform(0.0, 1.0),
                    device,
                )
                .argsort(0)
                .set_require_grad(false);

                perms_per_head.push(rand_indices);
                sign_per_head.push(
                    Tensor::<B, 1>::random([d], distribution, device)
                        .sign()
                        .set_require_grad(false)
                        .unsqueeze::<2>()
                        .unsqueeze::<3>()
                        .reshape([1, d, 1]),
                )
            });
            perms_per_encoder.push(Tensor::cat(perms_per_head, 0));
            sign_per_encoder.push(Tensor::cat(sign_per_head, 1));
        });

        Permuter {
            signs: sign_per_encoder,
            perms: perms_per_encoder,
            num_heads: self.num_heads,
        }
        .no_grad()
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
    pub fn forward(&self, x: Tensor<B, 3>, encoder_num: usize) -> Tensor<B, 3> {
        let out = self.permuter.forward(x, encoder_num);
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
