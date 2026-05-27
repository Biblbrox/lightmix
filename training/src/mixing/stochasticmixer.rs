use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    prelude::Tensor,
    tensor::{Distribution, Int, activation::softmax, backend::Backend, s},
};

use crate::mixing::sinkhorn;

#[derive(Module, Debug)]
pub struct StochasticMixer<B: Backend> {
    proj_qk_logits: Param<Tensor<B, 4>>, // [1, 2H, d, d] - fused qk scores
    proj_v: Linear<B>,                   // [E -> E] - unconstrained value projection
    temperature: f32,                    // sinkhorn temperature
    num_heads: usize,
    d: usize,
}

#[derive(Config, Debug)]
pub struct StochasticMixerConfig {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub temperature: f32,
}

/// One Sinkhorn iteration via two softmaxes — no loop needed.
/// Row softmax then col softmax approximates a doubly-stochastic matrix.
/// At low temperature both softmaxes become near-one-hot → near-permutation.
impl<B: Backend> StochasticMixer<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match B::ad_enabled(&x.device()) {
            true => self.forward_soft(x),
            false => self.forward_hard(x),
        }
    }

    /// Soft forward — doubly-stochastic Q/K, full differentiable attention.
    /// x: [B, N, E] → [B, N, E]
    pub fn forward_soft(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();
        let h = self.num_heads;
        let d = self.d;

        let x = x.reshape([b, n, h, d]).swap_dims(1, 2);
        let x_q = x.clone();
        let x_k = x.clone();

        let qk = sinkhorn(self.proj_qk_logits.val(), self.temperature);
        let w_q = qk.clone().slice_dim(1, s![0..h]); // [1, H, d, d]
        let w_k = qk.slice_dim(1, s![h..2 * h]); // [1, H, d, d]

        let q = x_q.matmul(w_q); // [B, H, N, d]
        let k = x_k.matmul(w_k); // [B, H, N, d]
        let v = self.proj_v.forward(x); // [B, H, N, d]

        // [B, H, N, d] @ [B, H, d, N] → [B, H, N, N]
        let p = softmax(q.matmul(k.transpose()) / (d as f32).sqrt(), 3).matmul(v);

        p.swap_dims(1, 2).reshape([b, n, e])
    }

    /// Hard forward — Q/K approximated as permutations via argmax.
    /// Q/K routing: pure feature indexing, no matmul.
    /// V aggregation: single gather, differentiable w.r.t. values.
    /// x: [B, N, E] → [B, N, E]
    pub fn forward_hard(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();
        let h = self.num_heads;
        let d = self.d;

        // argmax over input dim: [H, d, d] → [H, d, 1] → squeeze → [H, d]
        let qk = self.proj_qk_logits.val();

        // [1, H, 1, d]
        let qk = qk.argmax(2);
        let perm_q = qk.clone().slice_dim(1, s![0..h]);
        let perm_k = qk.slice_dim(1, s![h..2 * h]);

        let x = x.reshape([b, n, h, d]).swap_dims(1, 2); // [B, H, N, d]

        let perm_qk = Tensor::cat(vec![perm_q, perm_k], 3) // [1, H, 1, 2d]
            .expand([b, h, n, 2 * d]); // [B, H, N, 2d]
        let qk_out = x.clone().gather(3, perm_qk); // [B, H, N, 2d]
        let q = qk_out.clone().slice_dim(3, s![0..d]);
        let k = qk_out.slice_dim(3, s![d..2 * d]);

        // V: unconstrained, full matmul, carries gradient
        let v = self.proj_v.forward(x); // [B, H, N, d]

        // [B, H, N, d] @ [B, H, d, N] → [B, H, N, N]
        // argmax: [B, H, N, N] → [B, H, N, 1] → squeeze → [B, H, N]
        let idx = q
            .matmul(k.transpose())
            .argmax(3) // [B, H, N, 1]
            .expand([b, h, n, d]); // [B, H, N, d]

        v.gather(2, idx).swap_dims(1, 2).reshape([b, n, e])
    }

    /// Precompute permutations once after training for use at inference.
    pub fn extract_permutations(&self) -> (Tensor<B, 3, Int>, Tensor<B, 3, Int>) {
        let h = self.num_heads;
        let d = self.d;

        let qk = self.proj_qk_logits.val();

        let perm_q = qk
            .clone()
            .slice([0..1, 0..h, 0..d, 0..d])
            .argmax(2)
            .squeeze_dim(0);

        let perm_k = qk
            .slice([0..1, h..(2 * h), 0..d, 0..d])
            .argmax(2)
            .squeeze_dim(0);

        (perm_q, perm_k)
    }

    /// Pure inference — permutations passed in from extract_permutations().
    /// No Sinkhorn, no matmul for Q/K routing.
    /// x: [B, N, E] → [B, N, E]
    pub fn forward_inference(
        &self,
        x: Tensor<B, 3>,
        perm_qk: Tensor<B, 4, Int>, // [B, H, N, 2*d]
    ) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();
        let h = self.num_heads;
        let d = self.d;

        let x = x.clone().reshape([b, n, h, d]).swap_dims(1, 2); // [B, H, N, d]

        let qk_out = x.clone().gather(3, perm_qk); // [B, H, N, 2d]
        let q = qk_out.clone().slice_dim(3, s![0..d]);
        let k = qk_out.slice_dim(3, s![d..2 * d]);

        let v = self.proj_v.forward(x); // [B, H, N, d]

        // argmax: [B, H, N, N] → [B, H, N, 1] → squeeze → expand
        let idx = q
            .matmul(k.transpose())
            .argmax(3) // [B, H, N, 1]
            .expand([b, h, n, d]); // [B, H, N, d]

        v.gather(2, idx).swap_dims(1, 2).reshape([b, n, e])
    }
}

impl StochasticMixerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StochasticMixer<B> {
        assert!(
            self.embed_dim.is_multiple_of(self.num_heads),
            "embed_dim {} must be divisible by num_heads {}",
            self.embed_dim,
            self.num_heads
        );

        let d = self.embed_dim / self.num_heads;
        let h = self.num_heads;
        let logit_std = (1.0 / d as f64).sqrt();

        StochasticMixer {
            proj_qk_logits: Param::from_tensor(Tensor::<B, 4>::random(
                [1, 2 * h, d, d],
                Distribution::Normal(0.0, logit_std),
                device,
            ))
            .set_require_grad(true),
            proj_v: LinearConfig::new(d, d).init(device),
            temperature: self.temperature,
            num_heads: h,
            d,
        }
    }
}
