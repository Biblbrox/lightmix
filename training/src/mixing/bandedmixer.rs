use burn::{
    config::Config,
    module::{Module, Param},
    nn::LinearConfig,
    prelude::Tensor,
    tensor::{Int, activation::softmax, backend::Backend, ops::PadMode},
};

use crate::linear::{LinearLayer, monarch::MonarchLinearConfig};

#[derive(Config, Debug)]
pub struct BandedMixerConfig {
    pub embed_dim: usize,
    pub seq_length: usize,
    pub num_heads: usize,
    pub kernel_size: usize,
    pub temperature: f32,
    #[config(default = 4)]
    pub qk_shrink: usize,
    #[config(default = true)]
    pub use_monarch: bool,
}

#[derive(Module, Debug)]
pub struct BandedMixer<B: Backend> {
    proj_qkv: LinearLayer<B>,
    inv_scale: f32,
    band_bias: Param<Tensor<B, 4>>, // [H, N, 2w+1]
    temperature: f32,
    half_width: usize,
    num_heads: usize,
    tok_idx: Tensor<B, 4, Int>,
    dk_qk: usize,
    dk_v: usize,
}

impl<B: Backend> BandedMixer<B> {
    pub fn forward_hard(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();

        let h = self.num_heads;
        let dk_qk = self.dk_qk;
        let dk_v = self.dk_v;
        let w = self.half_width;
        let bw = 2 * w + 1;

        let qkv = self
            .proj_qkv
            .forward(x)
            .reshape([b, n, h, 2 * dk_qk + dk_v]); // [B,N,H,feat] — no swap_dims

        let q = qkv.clone().slice([0..b, 0..n, 0..h, 0..dk_qk]); // [B,N,H,dk_qk]
        let k = qkv.clone().slice([0..b, 0..n, 0..h, dk_qk..2 * dk_qk]);
        let v = qkv.slice([0..b, 0..n, 0..h, 2 * dk_qk..2 * dk_qk + dk_v]);

        let kv_pad = Tensor::cat(vec![k, v], 3)
            .pad([(0, 0), (w, w), (0, 0), (0, 0)], PadMode::Constant(0.0));

        let k_pad = kv_pad.clone().slice([0..b, 0..n + 2 * w, 0..h, 0..dk_qk]);
        let v_pad = kv_pad.slice([0..b, 0..n + 2 * w, 0..h, dk_qk..dk_qk + dk_v]);

        let k_win: Tensor<B, 5> = k_pad.unfold(1, bw, 1); // [B,N,H,dk_qk,bw]

        let idx = q
            .unsqueeze_dim(3) // [B,N,H,1,dk_qk]
            .matmul(k_win) // [B,N,H,1,bw]
            .squeeze_dim(3) // [B,N,H,bw]
            .mul_scalar(self.inv_scale)
            .add(self.band_bias.val()) // [B,N,H,bw] + [1,N,H,bw]
            .argmax(3); // [B,N,H,1]

        let pos = (idx + self.tok_idx.clone().expand([b, n, h, 1])).expand([b, n, h, dk_v]); // [B,N,H,dk_v]

        v_pad
            .gather(1, pos) // [B,N,H,dk_v] — dim 1 is N
            .reshape([b, n, e])
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();
        let h = self.num_heads;
        let dk_qk = self.dk_qk;
        let dk_v = self.dk_v;
        let w = self.half_width;
        let bw = 2 * w + 1;

        let qkv = self
            .proj_qkv
            .forward(x)
            .reshape([b, n, h, 2 * dk_qk + dk_v]);

        let q = qkv.clone().slice([0..b, 0..n, 0..h, 0..dk_qk]); // [B,N,H,dk_qk]
        let k = qkv.clone().slice([0..b, 0..n, 0..h, dk_qk..2 * dk_qk]);
        let v = qkv.slice([0..b, 0..n, 0..h, 2 * dk_qk..2 * dk_qk + dk_v]);

        let kv_pad = Tensor::cat(vec![k, v], 3)
            .pad([(0, 0), (w, w), (0, 0), (0, 0)], PadMode::Constant(0.0));

        let k_pad = kv_pad.clone().slice([0..b, 0..n + 2 * w, 0..h, 0..dk_qk]);
        let v_pad = kv_pad.slice([0..b, 0..n + 2 * w, 0..h, dk_qk..dk_qk + dk_v]);

        let k_win: Tensor<B, 5> = k_pad.unfold(1, bw, 1); // [B,N,H,dk_qk,bw]
        let v_win: Tensor<B, 5> = v_pad.unfold(1, bw, 1); // [B,N,H,dk_v,bw]

        let scores = q
            .unsqueeze_dim(3)
            .matmul(k_win) // [B,N,H,1,bw]
            .squeeze_dim::<4>(3) // [B,N,H,bw]
            .mul_scalar(self.inv_scale)
            .add(self.band_bias.val());

        let p = softmax(scores, 3); // [B,N,H,bw]

        v_win
            .matmul(p.unsqueeze_dim(3).transpose()) // [B,N,H,dk_v,1]
            .squeeze_dim::<4>(4) // [B,N,H,dk_v]
            .reshape([b, n, e])
    }
}

impl BandedMixerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BandedMixer<B> {
        let w = (self.kernel_size - 1) / 2;
        let window = 2 * w + 1;
        let dk_qk = self.embed_dim / self.num_heads / self.qk_shrink; // half head dim for Q,K
        let dk_v = self.embed_dim / self.num_heads; // full head dim for V

        let make_linear = |in_f: usize, out_f: usize| -> LinearLayer<B> {
            if self.use_monarch {
                LinearLayer::Monarch(MonarchLinearConfig::new(in_f, out_f).init(device))
            } else {
                LinearLayer::Dense(LinearConfig::new(in_f, out_f).init(device))
            }
        };

        BandedMixer {
            band_bias: Param::from_tensor(
                Tensor::<B, 3>::zeros([self.seq_length, self.num_heads, window], device)
                    .unsqueeze_dim(0), // [1, N, H, bw]
            )
            .set_require_grad(true),
            temperature: self.temperature,
            half_width: w,
            num_heads: self.num_heads,
            tok_idx: Tensor::<B, 1, Int>::arange(0..self.seq_length as i64, device).reshape([
                1,
                self.seq_length,
                1,
                1,
            ]),
            proj_qkv: make_linear(self.embed_dim, self.num_heads * dk_qk * 2 + self.embed_dim),
            dk_qk,
            dk_v,
            inv_scale: 1.0 / ((dk_qk as f32).sqrt() * self.temperature),
        }
    }
}
