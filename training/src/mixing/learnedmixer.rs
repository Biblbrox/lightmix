use std::sync::Arc;

use burn::{
    module::{Module, Param}, nn::{
        Linear, LinearConfig,
    }, prelude::*, tensor::Distribution
};

/// Permuter implementation with permutation matrix
///
/// * `signs`: [TODO:parameter]
/// * `perms`: [TODO:parameter]
/// * `num_heads`: [TODO:parameter]
#[derive(Module, Debug)]
pub struct LearnedPermuter<B: Backend> {
    signs: Tensor<B, 3>,
    sinkhorn_scores:  Param<Tensor<B, 3>>,  // [H, Nd, Nd]
    //perms: Tensor<B, 1, Int>,
    twiddle_left: Param<Tensor<B, 3>>,
    twiddle_right: Param<Tensor<B, 3>>,
    num_heads: usize,
    embed_dim: usize,
    seq_length: usize,
    pad_length: usize,  // Nd = next_power_of_two(seq_length)
    stage:      usize,
    sinkhorn_iters:   usize,
    temperature: f32,
    linear: Linear<B>
}

#[derive(Config, Debug)]
pub struct LearnedPermuterConfig {
    embed_dim: usize,
    seq_length: usize,
    num_heads: usize,
    out_channels: usize,
    num_encoders: usize,
    stage: usize,
    #[config(default = 20)]
    sinkhorn_iters: usize,
    temperature: f32,
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
            p = p.clone() / p.clone().sum_dim(2);  // row-normalise -> rows sum to 1
            p = p.clone() / p.clone().sum_dim(1);  // col-normalise -> cols sum to 1
        }

        //info!("{p}");
        p  // [H, Nd, Nd]
    }

    /// Butterfly mix at stride s = 2^stage
    ///
    ///   Layout after reshape to [B, H, n_blocks, 2s, E]:
    ///     left  = tokens [0 .. s)   within each block
    ///     right = tokens [s .. 2s)  within each block
    ///
    ///   Butterfly step:  y_left  = left + w * right   (a + w*b)
    ///               y_right = left - w * right   (a - w*b)
    fn butterfly_mix(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, _h_nd, dim] = x.dims();
        let nd = self.pad_length;
        let h  = self.num_heads;
        let s  = 1usize << self.stage;  // stride for this layer
        let n_blocks = nd / (2 * s);    // number of independent butterfly blocks

        debug_assert_eq!(nd % (2 * s), 0,
            "pad_length {nd} must be divisible by 2*stride {} (stage {})",
            2 * s, self.stage);

        let x = x.reshape([batch, h, nd, dim]);

        let x_blocks = x.reshape([batch, h, n_blocks, 2 * s, dim]);

        // Split at the stride boundary — no data movement, just views
        let left  = x_blocks.clone().narrow(3, 0, s);  // [B, H, n_blocks, s, E]
        let right = x_blocks.narrow(3, s, s);           // [B, H, n_blocks, s, E]

        let w_left = self.twiddle_left.val().reshape([1, h, n_blocks, s, dim]).cos();
        let w_right = self.twiddle_right.val().reshape([1, h, n_blocks, s, dim]).cos();

        let y_left  = left.clone() + w_left * right.clone(); // a + w*b
        let y_right = left          - w_right          * right;        // a − w*b

        // Reconstruct: [B, H, n_blocks, 2s, E] -> [B, H*Nd, E]
        Tensor::cat(vec![y_left, y_right], 3)
            .reshape([batch, h * nd, dim])
    }

    fn hard_permute(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.num_heads;

        // Converge P then take argmax per row → hard assignment index per head
        let p    = self.sinkhorn(self.sinkhorn_scores.val());       // [H, Nd, Nd]
        let perm = p.argmax(2).flatten::<1>(0, 2);        // [H*Nd]

        // Repeat x for each head then select
        x.repeat(&[1, h, 1])                               // [B, H*Nd, E]
         .select(1, perm)                                  // [B, H*Nd, E]
    } 

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // x: [B, N, E]
        let [b, _n, e] = x.dims();

        let pad   = self.pad_length - self.seq_length;
        let lpad  = pad / 2;
        let rpad  = pad - lpad;
        let x = Tensor::cat(vec![
            Tensor::<B, 3>::zeros([b, lpad, e], &x.device()),
            x.clone(),
            Tensor::<B, 3>::zeros([b, rpad, e], &x.device()),
        ], 1);
        // x: [B, Nd, E]

        //let x = x.clone().repeat_dim(1, self.num_heads);
        //let spectral = self.butterfly_mix(x.clone());
        //let permuted = spectral.select(1, self.perms.clone()) * self.signs.clone() + x.clone();
        //let permuted = permuted * self.signs.clone() + x;

        //permuted.swap_dims(1, 2)
        // out: [B, E, H*Nd]  — ready for downstream linear
        let x = x.repeat_dim(1, self.num_heads);             // [B, H*Nd, E]

        // Butterfly mix (learnable twiddle)
        //let spectral = self.butterfly_mix(x.clone());   // [B, H*Nd, E]
        let spectral = x.clone();   // [B, H*Nd, E]

        //let permuted = if self.sinkhorn_scores.is_require_grad() {
        let p    = self.sinkhorn(self.sinkhorn_scores.val());        // [H, Nd, Nd]
        let s4d  = spectral.clone().reshape([b, self.num_heads, self.pad_length, e]);
        let p4d  = p.unsqueeze_dim::<4>(0).repeat_dim(0, b);        // [B, H, Nd, Nd]
        let permuted = p4d.matmul(s4d).reshape([b, self.num_heads * self.pad_length, e]);
        //} else {
        //    self.hard_permute(spectral)
        //};

        let permuted = permuted * self.signs.clone();
        let permuted = self.butterfly_mix(permuted.clone());   // [B, H*Nd, E]

        // [B, E, H*Nd]
        let permuted = permuted.swap_dims(1, 2); 
        self.linear.forward(permuted).swap_dims(1, 2)
    }

}


impl LearnedPermuterConfig {
     pub fn init<B: Backend>(&self, device: &B::Device) -> LearnedPermuter<B> {
        let pad_length = self.seq_length.next_power_of_two();
        let nd         = pad_length;
        let max_stages = nd.trailing_zeros() as usize;

        assert!(
            self.stage < max_stages,
            "stage {} out of range — pad_length={nd} supports only {max_stages} stages (0..{})",
            self.stage, max_stages - 1
        );

        let mut signs_list   = Vec::<Tensor<B, 3>>::new();
        //let mut perms_list   = Vec::<Tensor<B, 1, Int>>::new();
        let mut scores_list  = Vec::<Tensor<B, 3>>::new();  // replaces perms_list
        let mut twiddle_list = Vec::<Tensor<B, 2>>::new();

        (0..self.num_heads).for_each(|_| {
            // Frozen random permutation — determines which originals get mixed
            //let rand_idx = Tensor::<B, 1>::random(
            //    Shape::new([nd]),
            //    Distribution::Uniform(0.0, 1.0),
            //    device,
            //).argsort(0);

            // Frozen ±1 signs — cheap nonlinearity before butterfly
            let signs = Tensor::<B, 2>::random(
                [nd, self.embed_dim],
                Distribution::Uniform(-1.0, 1.0),
                device,
            ).sign();

            // Learnable twiddle: initialise near 1 so first forward ≈ identity
            let twiddle = Tensor::<B, 2>::ones([nd / 2, self.embed_dim], device);

            let noise = Tensor::<B, 2>::random(
                [nd, nd],
                Distribution::Normal(0.0, 0.1),
                device,
            );
            // Approximate identity init: add a constant to the diagonal
            // Burn has no eye(), so build it as: arange outer-equal mask
            let idx   = Tensor::<B, 1, Int>::arange(0..nd as i64, device);
            let rows  = idx.clone().reshape([nd, 1]).float();
            let cols  = idx.reshape([1, nd]).float();
            let diag_mask = rows.equal(cols).float() * 3.0;   // +3 on diagonal
            let scores = diag_mask + noise;
            //let scores = Tensor::<B, 2>::random(
        //[nd,// nd],
            //    Distribution::Normal(0.0, 1.0),  // larger noise, no bias
            //device,
            //);

            //perms_list.push(rand_idx);
            signs_list.push(signs.unsqueeze_dim::<3>(0)); // [1, Nd, E]
            scores_list.push(scores.unsqueeze_dim::<3>(0));     // [1, Nd, Nd]
            twiddle_list.push(twiddle);
        });

        //let perms   = Tensor::cat(perms_list, 0);          // [H*Nd]
        let scores  = Tensor::cat(scores_list, 0);          // [H, Nd, Nd]
        let signs   = Tensor::cat(signs_list, 1);          // [1, H*Nd, E]
        let twiddle = Tensor::stack::<3>(twiddle_list, 0); // [H, Nd/2, E]
        
        LearnedPermuter {
            signs,
            //perms,
            sinkhorn_scores: Param::from_tensor(scores).set_require_grad(true),
            twiddle_left: Param::from_tensor(twiddle.clone()).set_require_grad(true),
            twiddle_right: Param::from_tensor(twiddle).set_require_grad(true),
            num_heads:  self.num_heads,
            embed_dim:  self.embed_dim,
            seq_length: self.seq_length,
            pad_length,
            stage: self.stage,
            sinkhorn_iters: self.sinkhorn_iters,
            temperature: self.temperature,
            linear: LinearConfig::new(pad_length * self.num_heads, self.seq_length).with_bias(false)
                .init(device),
        }
    }
}

