use burn::{
    module::{Module, Param},
    nn::{
        Linear, LinearConfig,
        conv::{Conv1d, Conv1dConfig},
    },
    prelude::*,
    tensor::Distribution,
};

#[derive(Config, Debug)]
pub struct BandedMixerConfig {
    pub embed_dim: usize,
    pub seq_length: usize,
    pub num_heads: usize,
    pub kernel_size: usize,
    #[config(default = 20)]
    pub sinkhorn_iters: usize,
    pub temperature: f32,
}

#[derive(Module, Debug)]
pub struct BandedMixer<B: Backend> {
    conv_qkv: Conv1d<B>,
    band_bias: Param<Tensor<B, 4>>, // [H, N, 2w+1]
    signs: Tensor<B, 3>,            // [N, E] frozen
    linear: Linear<B>,              // token mixer (like in MLPMixer)
    sinkhorn_iters: usize,
    temperature: f32,
    half_width: usize,
    num_heads: usize,
    tok_idx: Tensor<B, 3, Int>,
}

impl<B: Backend> BandedMixer<B> {
    //pub fn precompute_permutation(mut self) -> Self {
    //    let indices = self.sinkhorn(self.sinkhorn_scores.val());
    //    self.sinkhorn_matrix = Some(p);
    //    self
    //}

    fn learned_permut(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [b, n, e] = x.dims();
        let h = self.num_heads;
        let dk = e / h;

        // Q K V via depthwise conv
        let x_t = x.clone().swap_dims(1, 2); // [B, E, N]
        let qkv = self
            .conv_qkv
            .forward(x_t)
            .reshape([b, h, 3 * dk, n])
            .swap_dims(2, 3); // [B, H, N, dk * 3]
        let q = qkv.clone().slice([0..b, 0..h, 0..n, 0..dk]);
        let k = qkv.clone().slice([0..b, 0..h, 0..n, dk..2 * dk]);
        let v = qkv.slice([0..b, 0..h, 0..n, 2 * dk..3 * dk]);

        // Banded scores [B, H, N, 2w+1]
        let scores = self.banded_scores(&q, &k);

        let p = self.banded_softmax(scores); // [B, H, N, 2w+1]
        (p, v)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();
        let h = self.num_heads;
        let dk = e / h;

        let (p, v) = self.learned_permut(x);
        let routed = if self.band_bias.is_require_grad() {
            self.banded_matvec(&p, &v) // [B, H, N, dk]
        } else {
            let offsets = p.argmax(3).squeeze_dim::<3>(3); // [B, H, N]
            self.hard_gather(&v, offsets, n, dk) // [B, H, N, dk]
        };

        let routed = routed
            .swap_dims(1, 2) // [B, N, H, dk]
            .reshape([b, n, e]); // [B, N, E]

        let mixed = routed * self.signs.clone(); // [B, N, E]

        self.linear.forward(mixed.transpose()).transpose()
    }

    /// Banded Q·K^T scores, O(N*w*dk).
    fn banded_scores(&self, q: &Tensor<B, 4>, k: &Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, h, n, dk] = q.dims();
        let w = self.half_width;

        // Pad K symmetrically so every query position has 2w+1 key neighbours
        let zeros = |len: usize| Tensor::<B, 4>::zeros([b, h, len, dk], &k.device());
        let k_pad = Tensor::cat(vec![zeros(w), k.clone(), zeros(w)], 2); // [B,H,N+2w,dk]

        // unfold dim=2, size=2w+1, step=1 -> [B, H, N, dk, 2w+1]
        let k_win: Tensor<B, 5> = k_pad.unfold(2, 2 * w + 1, 1); // [B, H, N, dk, 2w+1]

        // q: [B,H,N,dk] -> [B,H,N,1,dk]
        let q_exp = q.clone().unsqueeze_dim(3);

        // [B,H,N,1,dk] @ [B,H,N,dk,2w+1] -> [B,H,N,1,2w+1] -> squeeze
        let scores = q_exp.matmul(k_win)
            .squeeze_dim(3)                  // [B, H, N, 2w+1]
            / (dk as f32).sqrt();

        let bias = self.band_bias.val(); // [B, H, N, 2w+1]
        scores + bias
    }

    /// Row-softmax over the band dimension.
    fn banded_softmax(&self, scores: Tensor<B, 4>) -> Tensor<B, 4> {
        let max = scores.clone().max_dim(3); // [B, H, N, 1]  — stability
        let exp = ((scores - max) / self.temperature).exp();
        let sum = exp.clone().sum_dim(3).clamp_min(1e-8);
        exp / sum // [B, H, N, 2w+1]
    }

    /// Soft banded matrix-vector product: Y = P_band @ V, O(N·w·dk).
    fn banded_matvec(&self, p: &Tensor<B, 4>, v: &Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, h, n, dk] = v.dims();
        let w = self.half_width;
        let bw = 2 * w + 1;

        let zeros = |len: usize| Tensor::<B, 4>::zeros([b, h, len, dk], &v.device());
        let v_pad = Tensor::cat(vec![zeros(w), v.clone(), zeros(w)], 2); // [B,H,N+2w,dk]

        // [B, H, N, dk, 2w+1]
        let v_win: Tensor<B, 5> = v_pad.unfold(2, bw, 1);
        // [B, H, N, 2w+1, dk]
        let v_win = v_win.swap_dims(3, 4);

        // p: [B,H,N,2w+1] -> [B,H,N,1,2w+1]
        let p_exp = p.clone().unsqueeze_dim(3);

        // [B,H,N,1,2w+1] @ [B,H,N,2w+1,dk] -> [B,H,N,1,dk] -> squeeze
        p_exp.matmul(v_win).squeeze_dim(3) // [B, H, N, dk]
    }

    /// Hard gather: for each token, pick the one neighbour with highest weight.
    fn hard_gather(
        &self,
        v: &Tensor<B, 4>,
        offsets: Tensor<B, 3, Int>, // [B, H, N] — index into band [0, 2w]
        n: usize,
        dk: usize,
    ) -> Tensor<B, 4> {
        // global_j = i - w + offset, clamped to valid range
        let global_j = (self.tok_idx.clone() - self.half_width.clone() as i64 + offsets)
            .clamp(0, n as i64 - 1); // [B, H, N]

        // Expand to [B, H, N, dk] for gather along dim=2
        let indices = global_j.unsqueeze_dim::<4>(3).repeat_dim(3, dk); // [B, H, N, dk]

        v.clone().gather(2, indices) // [B, H, N, dk]
    }
}

impl BandedMixerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BandedMixer<B> {
        let w = (self.kernel_size - 1) / 2;
        let window = 2 * w + 1;

        let band_bias = Param::from_tensor(
            Tensor::<B, 3>::zeros([self.num_heads, self.seq_length, window], device)
                .unsqueeze_dim(0),
        )
        .set_require_grad(true);

        let signs = Tensor::<B, 2>::random(
            [self.seq_length, self.embed_dim],
            Distribution::Uniform(-1.0, 1.0),
            device,
        )
        .sign()
        .unsqueeze_dim(0);

        BandedMixer {
            conv_qkv: Conv1dConfig::new(self.embed_dim, self.embed_dim * 3, self.kernel_size)
                .with_padding(nn::PaddingConfig1d::Explicit(w))
                .with_groups(self.num_heads)
                .with_bias(false)
                .init(device),
            band_bias,
            signs,
            linear: LinearConfig::new(self.seq_length, self.seq_length)
                .with_bias(false)
                .init(device),
            sinkhorn_iters: self.sinkhorn_iters,
            temperature: self.temperature,
            half_width: w,
            num_heads: self.num_heads,
            tok_idx: Tensor::<B, 1, Int>::arange(0..self.seq_length as i64, &device)
                .reshape([1, 1, self.seq_length])
                .repeat_dim(1, self.num_heads), // i_idx: absolute position of each query token [B, H, N]
        }
    }
}
