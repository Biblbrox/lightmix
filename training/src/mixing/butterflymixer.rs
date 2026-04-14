use burn::{module::Module, prelude::*, tensor::activation::sigmoid};
use image::EncodableLayout;

/// Permuter implementation with permutation matrix
///
/// * `signs`: [TODO:parameter]
/// * `perms`: [TODO:parameter]
/// * `num_heads`: [TODO:parameter]
#[derive(Module, Debug)]
pub struct ButterflyMixer<B: Backend> {
    twiddles: Vec<Tensor<B, 4>>,
    num_stages: usize, // log2(pad_length) — all stages
    num_heads: usize,
    embed_dim: usize,
    seq_length: usize,
    pad_length: usize, // Nd = next_power_of_two(seq_length)
    stage: usize,
}

#[derive(Config, Debug)]
pub struct ButterflyMixerConfig {
    embed_dim: usize,
    seq_length: usize,
    num_heads: usize,
    num_encoders: usize,
    stage: usize,
}

impl<B: Backend> ButterflyMixer<B> {
    /// Butterfly mix at stride s = 2^stage
    ///
    ///   Layout after reshape to [B, n_blocks, 2s, E]:
    ///     left  = tokens [0 .. s)   within each block
    ///     right = tokens [s .. 2s)  within each block
    ///
    ///   Butterfly step:  y_left  = left + w * right   (a + w*b)
    ///                    y_right = left - w * right   (a - w*b)
    fn butterfly_stage(&self, x: Tensor<B, 3>, stage: usize) -> Tensor<B, 3> {
        let [batch, nd, dim] = x.dims();
        let s = 1usize << stage;
        let n_blocks = nd / (2 * s);

        let w = self.twiddles[stage].clone().slice([0..s, 0..dim]);

        // Stack normal and reversed into batch dim — single kernel call
        let x_rev = x.clone().flip([1]);
        let x_both = Tensor::cat(vec![x, x_rev], 0); // [2B, nd, dim]

        let out_both = self.butterfly_step(x_both, s, n_blocks, batch * 2, nd, dim, w); // [2B, nd, dim]

        // Split and average
        let out_fwd = out_both.clone().slice([0..batch, 0..nd, 0..dim]);
        let out_rev = out_both.slice([batch..2 * batch, 0..nd, 0..dim]);

        (out_fwd + out_rev) * 0.5
    }

    // Extracted butterfly arithmetic shared by both passes
    fn butterfly_step(
        &self,
        x: Tensor<B, 3>,
        s: usize,
        n_blocks: usize,
        batch: usize,
        nd: usize,
        dim: usize,
        w: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let x_blocks = x.reshape([batch, n_blocks, 2 * s, dim]);
        let left = x_blocks
            .clone()
            .slice([0..batch, 0..n_blocks, 0..s, 0..dim]);
        let right = x_blocks.slice([0..batch, 0..n_blocks, s..2 * s, 0..dim]);

        let y_left = left.clone() + w.clone() * right.clone();
        let y_right = left - w * right;

        Tensor::cat(vec![y_left, y_right], 2).reshape([batch, nd, dim])
    }

    fn butterfly_mix_all_stages(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut out = x;
        for stage in 0..self.num_stages {
            out = self.butterfly_stage(out, self.num_stages - stage - 1);
        }
        out
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();

        if !n.is_power_of_two() {
            let pad = self.pad_length - self.seq_length;

            let lpad = pad / 2;
            let rpad = pad - lpad;

            // Reflect padding: boundary tokens mirror into pad positions
            // instead of zeros — pad positions carry real signal into butterfly
            let left_pad = x.clone().slice([0..b, 0..lpad, 0..e]); // [B, lpad, E]
            let right_pad = x.clone().slice([0..b, n - rpad..n, 0..e]); // [B, rpad, E]

            let x_pad = Tensor::cat(vec![left_pad, x, right_pad], 1); // [B, Nd,
            let x_pad = self.butterfly_mix_all_stages(x_pad); // [B, N, E]
            return x_pad.slice([0..b, lpad..lpad + self.seq_length, 0..e]);
        } else {
            return self.butterfly_mix_all_stages(x);
        }
    }
}

impl ButterflyMixerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ButterflyMixer<B> {
        let pad_length = self.seq_length.next_power_of_two();
        let nd = pad_length;
        let max_stages = nd.trailing_zeros() as usize;

        assert!(
            self.stage < max_stages,
            "stage {} out of range — pad_length={nd} supports only {max_stages} stages (0..{})",
            self.stage,
            max_stages - 1
        );

        let twiddles: Vec<Tensor<B, 4>> = (0..max_stages)
            .map(|stage| {
                let s = 1usize << stage;
                let n_blocks = nd / (2 * s);

                // FFT twiddle factor for stage k, position j: cos(2π * j / 2^(k+1))
                let angles: Vec<f32> = (0..s)
                    .map(|j| {
                        let angle = 2.0 * std::f32::consts::PI * j as f32 / (2.0 * s as f32);
                        angle.cos()
                    })
                    .collect();
                // Broadcast across E channels
                let angles_tensor = Tensor::<B, 1>::from_floats(angles.as_slice(), device)
                    .unsqueeze_dim::<2>(1)
                    .reshape([s, 1])
                    .repeat_dim(1, self.embed_dim); // [s, E]

                // Convert cosine values to logit space so sigmoid recovers them
                // sigmoid(x) = v  =>  x = log(v / (1-v))
                let logits = angles_tensor.clone().clamp(0.01, 0.99);
                let logits: Tensor<B, 2> = logits.clone() / (1.0 - logits.clone());

                sigmoid(
                    logits
                        .clone()
                        .log()
                        .slice([0..s, 0..self.embed_dim])
                        .reshape([1, 1, s, self.embed_dim])
                        .repeat_dim(1, n_blocks),
                )
                //logits.log()
            })
            .collect();

        ButterflyMixer {
            twiddles,
            num_heads: self.num_heads,
            embed_dim: self.embed_dim,
            seq_length: self.seq_length,
            pad_length,
            stage: self.stage,
            num_stages: max_stages,
        }
    }
}
