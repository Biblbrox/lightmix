use std::marker::PhantomData;

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig},
    prelude::Backend,
    tensor::Distribution,
};

use crate::{
    mixing::learnedmixer::{LearnedPermuter, LearnedPermuterConfig},
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
pub struct FastEncoderLayer<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    mix_layer: LearnedPermuter<B>,
    norm1: DynamicERF<B>,
    norm2: DynamicERF<B>,
    dropout: Dropout,
    activation: Gelu,
    drop_path: DropPath<B>,
}

#[derive(Config, Debug)]
pub struct FastEncoderLayerConfig {
    seq_length: usize,
    embed_dim: usize,
    num_heads: usize,
    hidden_dim: usize,
    dropout: f64,
    activation: String,
    num_encoders: usize,
    #[config(default = 0.5)]
    drop_path_prob: f64,
    encoder: usize,
    sinkhorn_temp: f32,
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
    activation: String,
    sinkhorn_temp: f32,
}

impl<B: Backend> FastEncoderLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Mixing block with stochastic depth
        let mix_out = self
            .dropout
            .forward(self.mix_layer.forward(self.norm1.forward(x.clone())));
        let x = self.drop_path.forward(x, mix_out);

        // FFN block with stochastic depth
        let ff_out = self._ff_block(self.norm2.forward(x.clone()));
        self.drop_path.forward(x, ff_out)
    }

    pub fn _ff_block(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.dropout.forward(
            self.linear2.forward(
                self.dropout
                    .forward(self.activation.forward(self.linear1.forward(x))),
            ),
        )
    }
}

impl FastEncoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FastEncoderLayer<B> {
        FastEncoderLayer {
            linear1: LinearConfig::new(self.embed_dim, self.hidden_dim).init(device),
            linear2: LinearConfig::new(self.hidden_dim, self.embed_dim).init(device),
            mix_layer: LearnedPermuterConfig::new(
                self.embed_dim,
                self.seq_length,
                self.encoder,
                self.sinkhorn_temp,
            )
            .init(device),
            norm1: DynamicERFConfig::new(self.embed_dim).init(device),
            norm2: DynamicERFConfig::new(self.embed_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            drop_path: DropPath::new(self.drop_path_prob),
            activation: Gelu::new(),
        }
    }
}

impl<B: Backend> FastEncoder<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
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

        for i in 0..self.num_layers {
            layers.push(
                FastEncoderLayerConfig::new(
                    self.seq_length,
                    self.embed_dim,
                    self.num_heads,
                    self.hid_dim,
                    self.dropout,
                    self.activation.clone(),
                    self.num_layers,
                    i,
                    self.sinkhorn_temp,
                )
                .init(device),
            );
        }
        FastEncoder {
            encoder_layers: layers,
            norm: Option::None,
        }
    }
}
