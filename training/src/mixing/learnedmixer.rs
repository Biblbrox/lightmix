use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::Distribution,
};

/// Permuter implementation with learned permutation matrix
#[derive(Module, Debug)]
pub struct LearnedPermuter<B: Backend> {
    signs: Param<Tensor<B, 3>>,
    sinkhorn_scores: Param<Tensor<B, 3>>, // [H, Nd, Nd]
    num_heads: usize,
    embed_dim: usize,
    seq_length: usize,
    sinkhorn_iters: usize,
    temperature: f32,
    linear: Linear<B>,
    sinkhorn_matrix: Option<Tensor<B, 3>>,
}

#[derive(Config, Debug)]
pub struct LearnedPermuterConfig {
    embed_dim: usize,
    seq_length: usize,
    num_heads: usize,
    temperature: f32,
    #[config(default = 20)]
    sinkhorn_iters: usize,
}

impl<B: Backend> LearnedPermuter<B> {
    /// Sinkhorn normalization: iteratively row- and column-normalise exp(S)
    /// until it converges to a doubly-stochastic matrix.
    ///
    /// s: [H, Nd, Nd]  ->  P: [H, Nd, Nd]  (rows and cols sum to 1)
    fn sinkhorn(&self, s: Tensor<B, 3>) -> Tensor<B, 3> {
        // Subtract row-max before exp for numerical stability
        let s = (s.clone() - s.max_dim(2)) / self.temperature;
        let mut p = s.exp();

        for _ in 0..self.sinkhorn_iters {
            p = p.clone() / p.clone().sum_dim(2); // row-normalise -> rows sum to 1
            p = p.clone() / p.clone().sum_dim(1); // col-normalise -> cols sum to 1
        }

        p // [H, Nd, Nd]
    }

    pub fn precompute_permutation(mut self) -> Self {
        let p = self.sinkhorn(self.sinkhorn_scores.val());
        self.sinkhorn_matrix = Some(p);
        self
    }

    fn get_permutation(&self) -> Tensor<B, 3> {
        match &self.sinkhorn_matrix {
            Some(p) => p.clone(),
            None => self.sinkhorn(self.sinkhorn_scores.val()),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();
        let head_dim = self.embed_dim / self.num_heads;

        let x_heads = x
            .reshape([b, self.seq_length, self.num_heads, head_dim])
            .swap_dims(1, 2); // [B, H, Nd, E/H]

        let p_soft = self.get_permutation(); // [H, Nd, Nd]

        let permuted = if self.sinkhorn_scores.is_require_grad() {
            // Training: soft matmul, gradients flow through P
            let p4d = p_soft.unsqueeze_dim::<4>(0); //.repeat_dim(0, b); // [B, H, Nd, Nd]
            p4d.matmul(x_heads) // [B, H, Nd, E/H]
        } else {
            // Expand indices to [B, H, Nd] then [B, H, Nd, E/H] for gather
            let indices = p_soft
                .argmax(2) // [H, Nd, 1]  — Burn keeps the dim
                .squeeze_dim::<2>(2) // [H, Nd]     — rank 2
                .unsqueeze_dim::<3>(0) // [1, H, Nd]  — rank 3
                .repeat_dim(0, b) // [B, H, Nd]  — rank 3
                .unsqueeze_dim::<4>(3) // [B, H, Nd, 1] — rank 4
                .repeat_dim(3, head_dim); // [B, H, Nd, E/H]

            x_heads.gather(2, indices) // [B, H, Nd, E/H]
        };

        let signs =
            self.signs
                .val()
                .clone()
                .reshape([1, self.num_heads, self.seq_length, head_dim]);
        let permuted = permuted * signs;

        let permuted = permuted.swap_dims(1, 2).reshape([b, e, self.seq_length]);

        self.linear.forward(permuted).transpose()
    }
}

impl LearnedPermuterConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LearnedPermuter<B> {
        let nd = self.seq_length;

        let mut signs_list = Vec::<Tensor<B, 3>>::new();
        let mut scores_list = Vec::<Tensor<B, 3>>::new(); // replaces perms_list
        let head_dim = self.embed_dim / self.num_heads;

        (0..self.num_heads).for_each(|_| {
            // Frozen random permutation — determines which originals get mixed
            let distribution = Distribution::Uniform(-1.0, 1.0);
            let signs = Tensor::<B, 2>::random([nd, head_dim], distribution, device).sign();

            let noise = Tensor::<B, 2>::random([nd, nd], Distribution::Normal(0.0, 0.1), device);
            let idx = Tensor::<B, 1, Int>::arange(0..nd as i64, device);
            let rows = idx.clone().reshape([nd, 1]).float();
            let cols = idx.reshape([1, nd]).float();
            let diag_mask = rows.equal(cols).float() * 3.0; // +3 on diagonal
            let scores = diag_mask + noise;

            signs_list.push(signs.unsqueeze_dim::<3>(0)); // [1, Nd, E]
            scores_list.push(scores.unsqueeze_dim::<3>(0)); // [1, Nd, Nd]
        });

        let scores = Tensor::cat(scores_list, 0); // [H, Nd, Nd]

        let signs = Tensor::cat(signs_list, 0);

        LearnedPermuter {
            signs: Param::from_tensor(signs).set_require_grad(true),
            sinkhorn_scores: Param::from_tensor(scores).set_require_grad(true),
            num_heads: self.num_heads,
            embed_dim: self.embed_dim,
            seq_length: self.seq_length,
            sinkhorn_iters: self.sinkhorn_iters,
            temperature: self.temperature,
            linear: LinearConfig::new(self.seq_length, self.seq_length)
                .with_bias(false)
                .init(device),
            sinkhorn_matrix: None,
        }
    }
}
