use std::{ops::Add, sync::Arc};

use burn::{
    module::{Module, Param},
    nn::{
        Linear, LinearConfig,
        pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig, MaxPool1d, MaxPool1dConfig},
    },
    prelude::*,
    tensor::{Distribution, activation::gelu, module::adaptive_avg_pool1d},
};

use crate::norm::{DynamicERF, DynamicERFConfig};

/// Permuter implementation with indicies
///
/// * `signs`: [TODO:parameter]
/// * `perms`: [TODO:parameter]
/// * `num_heads`: [TODO:parameter]
#[derive(Module, Debug)]
pub struct WeightedPermuter<B: Backend> {
    //signs: Tensor<B, 3>,
    perms: Vec<Tensor<B, 1, Int>>,
    //perms_heads: Tensor<B, 1, Int>,
    params: Param<Tensor<B, 3>>,
    num_heads: usize,
    linear: Linear<B>,
    norm: DynamicERF<B>,
}

#[derive(Config, Debug)]
pub struct WeightedPermuterConfig {
    embed_dim: usize,
    seq_length: usize,
    num_heads: usize,
    out_channels: usize,
    num_encoders: usize,
    encoder: usize,
}

impl<B: Backend> WeightedPermuter<B> {
    //pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
    //    // [H * N, E]
    //    let perms_token = self.perms_tokens.clone();
    //    let perms_heads = self.perms_heads.clone();
    //    let signs = self.signs.clone();

    //    // [B, N * H, E]
    //    let x = x.select(1, perms_token);
    //    let x = x.select(1, perms_heads);

    //    // [B, N * H, E]
    //    let x = x * signs;

    //    self.linear2.forward(self.linear1.forward(x.swap_dims(-1, -2))).swap_dims(-1, -2)
    //}

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();

        //let cls   = x.clone().slice([0..b, 0..1, 0..e]);       // [B, 1, E]
        //let tokens = x.clone().select(1, self.perms[0].clone()) * self.params.val();
        //let tokens = tokens.clone() + tokens.clone().select(1, self.perms[1].clone()) * self.params.val();
        //let tokens = tokens.clone() + tokens.clone().select(1, self.perms[2].clone()) * self.params.val();
        //let tokens = tokens.clone() + tokens.clone().select(1, self.perms[3].clone()) * self.params.val();

        //let x_perm = self.perms.iter().fold(Tensor::zeros([b, n * self.num_heads, e], &x.device()), |acc, cur| {
        //    acc + x.clone().select(1, cur.clone())
        //});

        // 2. Cross-head shuffle: mix which head sees which token
        //let x_perm = x_perm
        //    .select(1, self.perms_heads.clone());

        let global = adaptive_avg_pool1d(x.clone().transpose(), self.num_heads); // [B, E, H]
        let global = burn::tensor::activation::gelu(global); // nonlinearity between layers
        //let x_mixed = x_mixed.transpose();
        //let x_mixed = Tensor::cat(vec![x_mixed, cls], 1);

        global.transpose().mean_dim(1)
            + x.clone()
            + x.clone().select(1, self.perms[0].clone())
            + x.clone().select(1, self.perms[1].clone()) * self.params.val()
            + x.clone().select(1, self.perms[1].clone()) * self.params.val()
            + x.clone().select(1, self.perms[3].clone()) * self.params.val()
            + x.clone().select(1, self.perms[2].clone()) * self.params.val()
            + x.clone().select(1, self.perms[3].clone()) * self.params.val()
            + x.clone().select(1, self.perms[2].clone()) * self.params.val()
            + x.clone().select(1, self.perms[1].clone()) * self.params.val()
            + x.clone().select(1, self.perms[3].clone()) * self.params.val()
            + x.clone().select(1, self.perms[1].clone()) * self.params.val()
            + x.clone().select(1, self.perms[3].clone()) * self.params.val()
        //self.norm.forward(x_mixed)
    }
}

impl WeightedPermuterConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WeightedPermuter<B> {
        let d = self.seq_length;
        let h = self.num_heads;

        let mut perms = vec![];
        for j in 0..4 {
            let indices = (0..d).map(|i| ((i + j) % d) as i32).collect();
            perms.push(Tensor::<B, 1, Int>::from_data(
                TensorData::new(indices, [d]),
                device,
            ));
        }

        //let perms_heads = Tensor::<B, 1, Int>::from_data(
        //    TensorData::new(
        //        (0..d)
        //            .map(|i| (i ^ (self.encoder + 1)) as i32)
        //            .collect(),
        //        [h * d]
        //    ),
        //    device,
        //);

        let params =
            Param::from_tensor(Tensor::ones([1, d, self.embed_dim], device)).set_require_grad(true);

        WeightedPermuter {
            //signs,
            //perms_heads,
            num_heads: self.num_heads,
            perms,
            //linear1: LinearConfig::new(self.seq_length * self.num_heads, self.seq_length * self.num_heads / (self.num_heads / 2)).init(device),
            //linear2: LinearConfig::new(self.seq_length * self.num_heads / (self.num_heads / 2), self.seq_length).init(device),
            linear: LinearConfig::new(d, self.num_heads).init(device),
            params,
            norm: DynamicERFConfig::new(self.embed_dim).init(device),
        }
    }
}
