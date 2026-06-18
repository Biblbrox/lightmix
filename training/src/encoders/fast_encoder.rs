use std::marker::PhantomData;

use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig},
    tensor::{Distribution, Shape, backend::Backend, ops::PadMode},
};

use crate::{
    attention::stochasticwindowmixer::{StochasticWindowMixer, StochasticWindowMixerConfig},
    linear::{LinearLayer, monarch::MonarchLinearConfig},
    norm::{DynamicERF, DynamicERFConfig},
};

#[derive(Module, Debug)]
pub struct DropPath<B: Backend> {
    drop_prob: f64,
    phantom: PhantomData<B>,
}

impl<B: Backend> DropPath<B> {
    pub fn new(drop_prob: f64) -> Self {
        Self {
            drop_prob,
            phantom: PhantomData,
        }
    }

    // Applies stochastic depth: randomly drops the residual branch per sample.
    // x:        the main path (before residual add)
    // residual: the branch output to stochastically drop
    pub fn forward(&self, x: Tensor<B, 3>, residual: Tensor<B, 3>) -> Tensor<B, 3> {
        // During inference or if drop_prob is 0 — passthrough
        if self.drop_prob == 0.0 || !B::ad_enabled(&x.device()) {
            return x + residual;
        }

        let [batch, _, _] = x.dims();
        let device = x.device();
        let keep_prob = 1.0 - self.drop_prob;

        // Per-sample binary mask: [B, 1, 1] — whole residual dropped per sample
        let mask =
            Tensor::<B, 3>::random([batch, 1, 1], Distribution::Bernoulli(keep_prob), &device)
                / keep_prob; // rescale so expectation is preserved

        x + residual * mask
    }
}

#[derive(Module, Debug)]
pub struct TokenMerger<B: Backend> {
    pos: Param<Tensor<B, 3>>,
    proj: Linear<B>,
    scale: Param<Tensor<B, 3>>,
}

#[derive(Config, Debug)]
pub struct TokenMergerConfig {
    embed_dim: usize,
    seq_length: usize,
}

impl TokenMergerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TokenMerger<B> {
        let out_seq = (self.seq_length / 2).max(1);

        TokenMerger {
            pos: Param::<Tensor<B, 3>>::from_tensor(Tensor::<B, 3>::zeros(
                Shape::new([1, out_seq, self.embed_dim]),
                device,
            ))
            .set_require_grad(true),
            proj: LinearConfig::new(self.embed_dim * 2, self.embed_dim)
                .with_bias(false)
                .init(device),
            scale: Param::from_tensor(Tensor::<B, 3>::zeros([1, 1, 1], device))
                .set_require_grad(true),
        }
    }
}

impl<B: Backend> TokenMerger<B> {
    /// x: [B, N, E] → [B, N/2, E]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();

        if n <= 1 {
            return x;
        }

        let half = n / 2;

        let dst = x.clone().slice([0..b, 0..half, 0..e]); // [B, N/2, E]
        let src = x.slice([0..b, half..n, 0..e]); // [B, N/2, E]

        self.proj
            .forward(Tensor::cat(vec![src.clone(), dst.clone()], 2))
            * self.scale.val()
            + (dst + src) / 2.0
            + self.pos.val()
    }
}

#[derive(Module, Debug)]
pub struct FastEncoderLayer<B: Backend> {
    linear1: LinearLayer<B>,
    linear2: LinearLayer<B>,
    mix_layer: StochasticWindowMixer<B>,
    norm1: DynamicERF<B>,
    norm2: DynamicERF<B>,
    dropout: Dropout,
    activation: Gelu,
    drop_path: DropPath<B>,

    merger: TokenMerger<B>,
}

#[derive(Config, Debug)]
pub struct FastEncoderLayerConfig {
    seq_length: usize,
    embed_dim: usize,
    num_heads: usize,
    hidden_dim: usize,
    dropout: f64,
    #[config(default = 0.0)]
    drop_path_prob: f64,
    encoder: usize,
    sinkhorn_temp: f32,
    #[config(default = true)]
    use_monarch: bool,
    grid_h: usize,
    grid_w: usize,
}

#[derive(Module, Debug)]
pub struct FastEncoder<B: Backend> {
    encoder_layers: Vec<FastEncoderLayer<B>>,
    norm: Option<DynamicERF<B>>,
}

#[derive(Config, Debug)]
pub struct FastEncoderConfig {
    num_layers: usize,
    grid_size: usize,
    seq_length: usize,
    embed_dim: usize,
    num_heads: usize,
    hid_dim: usize,
    dropout: f64,
    sinkhorn_temp: f32,
}

impl<B: Backend> FastEncoderLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Mixing block with stochastic depth
        let mix_out = self.mix_layer.forward(self.norm1.forward(x.clone()));
        let mix_out = self.dropout.forward(mix_out);
        let x = self.drop_path.forward(x, mix_out);

        // FFN block with stochastic depth
        let ff_out = self._ff_block(self.norm2.forward(x.clone()));

        self.drop_path.forward(x, ff_out)
        // Merge tokens N -> N/2 before the FFN
        //self.merger.forward(x)
    }

    pub fn _ff_block(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.linear1.forward(x);
        //let hidden = hidden.clamp(-1e4, 1e4);
        self.dropout.forward(
            self.linear2
                .forward(self.dropout.forward(self.activation.forward(hidden))),
        )
    }
}

impl FastEncoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FastEncoderLayer<B> {
        let make_linear = |in_f: usize, out_f: usize| -> LinearLayer<B> {
            if self.use_monarch {
                LinearLayer::Monarch(MonarchLinearConfig::new(in_f, out_f).init(device))
            } else {
                LinearLayer::Dense(LinearConfig::new(in_f, out_f).init(device))
            }
        };

        FastEncoderLayer {
            linear1: make_linear(self.embed_dim, self.hidden_dim),
            linear2: make_linear(self.hidden_dim, self.embed_dim),
            mix_layer: StochasticWindowMixerConfig::new(
                self.embed_dim,
                self.seq_length,
                self.num_heads,
                3,
                self.sinkhorn_temp,
            )
            .init(device),
            norm1: DynamicERFConfig::new(self.embed_dim).init(device),
            norm2: DynamicERFConfig::new(self.embed_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            drop_path: DropPath::new(self.drop_path_prob),
            activation: Gelu::new(),
            merger: TokenMergerConfig::new(self.embed_dim, self.seq_length).init(device),
        }
    }
}

impl<B: Backend> FastEncoder<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, n, _] = x.dims();

        // Pad the sequence to the next power of 2
        let pad = n.next_power_of_two() - n;
        let x = x.pad([(0, 0), (0, pad), (0, 0)], PadMode::Constant(0.0));

        let mut output = x.clone();

        for layer in self.encoder_layers.iter() {
            output = layer.forward(output);
        }

        if let Some(norm) = &self.norm {
            output = norm.forward(output);
        }

        output
    }
}

impl FastEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FastEncoder<B> {
        let mut layers = Vec::new();
        let seq_length = self.seq_length.next_power_of_two();

        for i in 0..self.num_layers {
            //let layer_seq = (seq_length >> i).max(1);
            let layer_seq = seq_length;
            layers.push(
                FastEncoderLayerConfig::new(
                    layer_seq,
                    self.embed_dim,
                    self.num_heads,
                    self.hid_dim,
                    self.dropout,
                    i,
                    self.sinkhorn_temp,
                    self.grid_size,
                    self.grid_size,
                )
                .with_drop_path_prob(((i + 1) as f64 / self.num_layers as f64) * 0.1)
                .with_use_monarch(false)
                .init(device),
            );
        }
        FastEncoder {
            encoder_layers: layers,
            norm: Option::None,
        }
    }
}
