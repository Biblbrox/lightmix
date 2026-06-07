use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    prelude::Tensor,
    tensor::{Distribution, Int, activation::softmax, backend::Backend, ops::PadMode, s},
};

use crate::mixing::sinkhorn;

#[derive(Config, Debug)]
pub struct StochasticWindowMixerConfig {
    pub embed_dim: usize,
    pub seq_length: usize,
    pub num_heads: usize,
    pub kernel_size: usize,
    pub temperature: f32,
}

#[derive(Module, Debug)]
pub struct StochasticWindowMixer<B: Backend> {
    proj_v: Linear<B>,
    proj_qk_logits: Param<Tensor<B, 4>>, // [1, 2H, d, d] - fused qk scores
    inv_scale: f32,
    band_bias: Param<Tensor<B, 4>>, // [H, N, 2w+1]
    temperature: f32,
    half_width: usize,
    num_heads: usize,
    tok_idx: Tensor<B, 4, Int>,
    dk: usize,
    window_indices: Tensor<B, 1, Int>, // [N * bw]
    win_idx_4d: Tensor<B, 4, Int>,
}

impl<B: Backend> StochasticWindowMixer<B> {
    pub fn pad(&self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let w = self.half_width;
        let k_pad = k.pad([(0, 0), (w, w), (0, 0), (0, 0)], PadMode::Constant(0.0));
        let v_pad = v.pad([(0, 0), (w, w), (0, 0), (0, 0)], PadMode::Constant(0.0));
        (k_pad, v_pad)
    }

    fn local_window(&self, x: Tensor<B, 4>) -> Tensor<B, 5> {
        let [b, n, h, dk] = x.dims();
        let bw = 2 * self.half_width + 1;

        let flat_idx = self.window_indices.clone();

        // select on dim 1: pulls whole [H, dk] rows by absolute position
        // [B, N, H, dk] → [B, N*bw, H, dk]
        let gathered = x.select(1, flat_idx);

        // Restore window structure and move bw to the last dim
        gathered
            .reshape([b, n, bw, h, dk]) // [B, N, bw, H, dk]
            .swap_dims(2, 3) // [B, N, H, bw, dk]
            .transpose() // [B, N, H, dk, bw]
    }

    //pub fn forward_hard(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
    //    let [b, n, e] = x.dims();

    //    let dk = self.dk;
    //    let w = self.half_width;
    //    let h = self.num_heads;
    //    let bw = 2 * w + 1;

    //    let qk = self.proj_qk_logits.val();

    //    let qk = qk.argmax(2); // [B, N, 2*H, d]
    //    let perm_q = qk.clone().slice_dim(1, s![0..n]);
    //    let perm_k = qk.slice_dim(1, s![n..2 * n]);

    //    // [B, H, N, d]
    //    let x = x.reshape([b, n, h, dk]);

    //    let perm_qk = Tensor::cat(vec![perm_q, perm_k], 3).expand([b, n, h, 2 * dk]);
    //    let qk_out = x.clone().gather(3, perm_qk); // [B, H, N, 2d]
    //    // [B, H]
    //    let q = qk_out.clone().slice_dim(3, s![0..dk]).unsqueeze_dim(3);
    //    let k = qk_out.slice_dim(3, s![dk..2 * dk]);
    //    let v = self.proj_v.forward(x); // [B, N, H, d]

    //    let (k_pad, v_pad) = self.pad(k, v);
    //    //let v_pad = v.pad([(0, 0), (w, w), (0, 0), (0, 0)], PadMode::Constant(0.0));

    //    let k_win: Tensor<B, 5> = k_pad.unfold(1, bw, 1); // [B,N,H,dk,bw]
    //    //let k_win = self.local_window(k);

    //    let idx =
    //        (q.matmul(k_win).squeeze_dim(3) * self.inv_scale + self.band_bias.val()).argmax(3);

    //    let pos = idx + self.tok_idx.clone(); // [B,N,H,dk]

    //    v_pad.gather(1, pos).reshape([b, n, e])
    //}

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match B::ad_enabled(&x.device()) {
            true => self.forward_soft(x),
            false => self.forward_hard(x),
        }
    }

    pub fn forward_hard(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();
        let dk = self.dk;
        let h = self.num_heads;
        let bw = 2 * self.half_width + 1;

        let qk = self
            .proj_qk_logits
            .val()
            .argmax(2)
            .expand([b, 2 * n, h, dk]);
        let perm_q = qk.clone().slice_dim(1, s![0..n]);
        let perm_k = qk.slice_dim(1, s![n..2 * n]);

        let x = x.reshape([b, n, h, dk]);

        let q = x.clone().gather(3, perm_q).unsqueeze_dim(3);
        let k = x.clone().gather(3, perm_k);
        let v = self.proj_v.forward(x);

        let k_win = self.local_window(k);

        let best_i =
            (q.matmul(k_win).squeeze_dim(3) * self.inv_scale + self.band_bias.val()).argmax(3); // [B, N, H]

        // [N*bw] -> [1, N, bw] -> [B, N, H, bw], gather best slot -> [B, N, H]
        let abs_pos = self
            .win_idx_4d
            .clone()
            .expand([b, n, h, bw])
            .gather(3, best_i) // [B, N, H, 1]
            .expand([b, n, h, dk]);

        // gather directly from unpadded v — no pad, no tok_idx
        v.gather(1, abs_pos).reshape([b, n, e])
    }

    pub fn forward_soft(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();
        let dk = self.dk;
        let h = self.num_heads;

        let x = x.reshape([b, n, h, dk]); // [B, H, N, d]

        let qk = sinkhorn(self.proj_qk_logits.val(), self.temperature);

        let w_q = qk.clone().slice_dim(1, s![0..n]); // [1, N, d, d]
        let w_k = qk.slice_dim(1, s![n..2 * n]); // [1, N, d, d]

        let q = x.clone().matmul(w_q).unsqueeze_dim(3); // [B, N, H, 1, d]
        let k = x.clone().matmul(w_k); // [B, N, H, d]
        let v = self.proj_v.forward(x); // [B, N, H, d]

        let k_win = self.local_window(k);
        let v_win = self.local_window(v);

        let scores = q.matmul(k_win).squeeze_dim(3) * self.inv_scale + self.band_bias.val();

        let p = softmax(scores, 3);

        let out = v_win.matmul(p.unsqueeze_dim(4));

        out.reshape([b, n, e])
    }
}

impl StochasticWindowMixerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StochasticWindowMixer<B> {
        let w = (self.kernel_size - 1) / 2;
        let window = 2 * w + 1;
        let dk = self.embed_dim / self.num_heads; // head dim
        let n = self.seq_length;

        let logit_std = (1.0 / dk as f64).sqrt();

        let pos = Tensor::<B, 1, Int>::arange(0..n as i64, device).reshape([1, n, 1, 1]);
        let offsets = Tensor::<B, 1, Int>::arange(-(w as i64)..(w as i64 + 1), device)
            .reshape([1, 1, 1, window]);
        let window_indices = (pos + offsets).clamp(0, n as i64 - 1); // [1, N, 1, bw]

        StochasticWindowMixer {
            band_bias: Param::from_tensor(Tensor::<B, 4>::zeros(
                [1, self.seq_length, self.num_heads, window],
                device,
            ))
            .set_require_grad(true),
            temperature: self.temperature,
            half_width: w,
            num_heads: self.num_heads,
            proj_qk_logits: Param::from_tensor(Tensor::<B, 4>::random(
                [1, 2 * self.seq_length, dk, dk],
                Distribution::Normal(0.0, logit_std),
                device,
            ))
            .set_require_grad(true),
            tok_idx: Tensor::<B, 1, Int>::arange(0..self.seq_length as i64, device)
                .reshape([1, self.seq_length, 1, 1])
                .expand([1, self.seq_length, self.num_heads, dk]),
            proj_v: LinearConfig::new(dk, dk).init(device),
            dk,
            inv_scale: 1.0 / ((dk as f32).sqrt() * self.temperature),
            window_indices: window_indices.clone().reshape([n * window]),
            win_idx_4d: window_indices.reshape([1, n, 1, window]).expand([
                1,
                n,
                self.num_heads,
                window,
            ]),
        }
    }
}
