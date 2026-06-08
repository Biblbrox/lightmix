use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    prelude::Tensor,
    tensor::{Distribution, Int, activation::softmax, backend::Backend, ops::PadMode, s},
};

use crate::mixing::sinkhorn;

#[derive(Config, Debug)]
pub enum StochasticMode {
    Q,
    K,
    Qk,
    Qkv,
}

impl StochasticMode {
    /// How many N-blocks the fused logit tensor needs.
    /// Layout: [Q block | K block | V block], each of size [N, dk, dk].
    fn num_blocks(&self) -> usize {
        match self {
            StochasticMode::Q => 1,
            StochasticMode::K => 1,
            StochasticMode::Qk => 2,
            StochasticMode::Qkv => 3,
        }
    }
}

#[derive(Config, Debug)]
pub struct StochasticWindowMixerConfig {
    pub embed_dim: usize,
    pub seq_length: usize,
    pub num_heads: usize,
    pub kernel_size: usize,
    pub temperature: f32,
    #[config(default = "StochasticMode::Qk")]
    pub mode: StochasticMode,
}

#[derive(Module, Debug)]
pub struct StochasticWindowMixer<B: Backend> {
    proj_v: Linear<B>,
    /// Fused logit tensor. Size along dim 1 is mode.num_blocks() * N.
    /// Q block: [0..N], K block: [N..2N], V block (Qkv only): [2N..3N].
    /// For Q-only mode: only [0..N] is used (K = identity).
    /// For K-only mode: only [0..N] is used (Q = identity), interpreted as K.
    proj_logits: Param<Tensor<B, 4>>, // fused qk scores
    inv_scale: f32,
    band_bias: Param<Tensor<B, 4>>, // [H, N, 2w+1]
    temperature: f32,
    half_width: usize,
    num_heads: usize,
    tok_idx: Tensor<B, 4, Int>,
    dk: usize,
    window_indices: Tensor<B, 1, Int>, // [N * bw]
    win_idx_4d: Tensor<B, 4, Int>,
    mode: StochasticMode,
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
            .permute([0, 1, 3, 4, 2]) // [B, N, H, dk, bw]
    }

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

        let qk = self.proj_logits.val().argmax(2).expand([b, 2 * n, h, dk]);
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

        let qk = sinkhorn(self.proj_logits.val(), self.temperature);

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
            proj_logits: Param::from_tensor(Tensor::<B, 4>::random(
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
            mode: StochasticMode::Qk,
        }
    }
}

/// Since we use select instead of unfold as a hack to make window creation faster,
/// we must be sure that it still produces the same results as unfold
#[cfg(test)]
mod tests {
    use burn::{
        backend::{Flex, flex::FlexDevice},
        tensor::{Int, Shape, Tensor, TensorData, ops::PadMode, s},
    };

    type B = Flex;
    type TestDevice = FlexDevice;

    fn device() -> TestDevice {
        Default::default()
    }

    /// Build window_indices (flat [N*bw]) the same way init() does.
    fn make_window_indices(n: usize, w: usize, device: &TestDevice) -> Tensor<B, 1, Int> {
        let window = 2 * w + 1;
        let pos = Tensor::<B, 1, Int>::arange(0..n as i64, device).reshape([1, n, 1, 1]);
        let offsets = Tensor::<B, 1, Int>::arange(-(w as i64)..(w as i64 + 1), device)
            .reshape([1, 1, 1, window]);
        (pos + offsets).clamp(0, n as i64 - 1).reshape([n * window])
    }

    /// select-based local_window: drop-in replacement for pad+unfold.
    /// x: [B, N, H, dk] → [B, N, H, dk, bw]
    fn local_window_select(
        x: Tensor<B, 4>,
        window_indices: Tensor<B, 1, Int>,
        w: usize,
    ) -> Tensor<B, 5> {
        let [b, n, h, dk] = x.dims();
        let bw = 2 * w + 1;
        let gathered = x.select(1, window_indices); // [B, N*bw, H, dk]
        gathered
            .reshape([b, n, bw, h, dk])
            .swap_dims(2, 3) // [B, N, H, bw, dk]
            .transpose() // [B, N, H, dk, bw]
    }

    /// Original pad+unfold approach.
    /// x: [B, N, H, dk] → [B, N, H, dk, bw]
    fn local_window_unfold(x: Tensor<B, 4>, w: usize) -> Tensor<B, 5> {
        let bw = 2 * w + 1;
        let x_pad = x.pad([(0, 0), (w, w), (0, 0), (0, 0)], PadMode::Constant(0.0));
        x_pad.unfold(1, bw, 1) // [B, N, H, dk, bw]
    }

    fn max_abs_diff(a: Tensor<B, 5>, b: Tensor<B, 5>) -> f32 {
        (a - b).abs().max().into_scalar()
    }

    /// Test 1: Interior tokens — both approaches must agree exactly.
    /// Uses a window that never touches the boundary so clamping vs zero-padding
    /// doesn't matter. This is the pure correctness test.
    #[test]
    fn test_interior_tokens_match() {
        let device = device();
        let (b, n, h, dk, w) = (2, 8, 2, 4, 1); // w=1 → bw=3, interior = tokens 1..7
        let bw = 2 * w + 1;

        // Deterministic input: value = position index so we can reason about it
        let data: Vec<f32> = (0..(b * n * h * dk) as i32).map(|i| i as f32).collect();
        let x =
            Tensor::<B, 4>::from_data(TensorData::new(data, Shape::new([b, n, h, dk])), &device);

        let win_idx = make_window_indices(n, w, &device);
        let select_out = local_window_select(x.clone(), win_idx, w);
        let unfold_out = local_window_unfold(x, w);

        assert_eq!(select_out.dims(), [b, n, h, dk, bw]);
        assert_eq!(unfold_out.dims(), [b, n, h, dk, bw]);

        // Slice out interior tokens only (skip first and last w tokens)
        let select_interior = select_out.slice_dim(1, s![w..n - w]);
        let unfold_interior = unfold_out.slice_dim(1, s![w..n - w]);

        let diff = max_abs_diff(select_interior, unfold_interior);
        assert!(diff < 1e-5, "Interior tokens differ: max abs diff = {diff}");
    }

    /// Test 2: Boundary behaviour difference is expected and documented.
    /// select uses clamp (repeats edge token), unfold uses zero padding.
    /// This test asserts they DIFFER at boundaries (confirming our understanding)
    /// and that the difference is exactly at the boundary tokens.
    #[test]
    fn test_boundary_behaviour_difference() {
        let device = device();
        let (b, n, h, dk, w) = (1, 6, 1, 2, 2); // w=2 → first/last 2 tokens are boundary

        let data: Vec<f32> = (0..(b * n * h * dk) as i32)
            .map(|i| i as f32 + 1.0)
            .collect();
        let x =
            Tensor::<B, 4>::from_data(TensorData::new(data, Shape::new([b, n, h, dk])), &device);

        let win_idx = make_window_indices(n, w, &device);
        let select_out = local_window_select(x.clone(), win_idx, w);
        let unfold_out = local_window_unfold(x, w);

        // Interior tokens [w..n-w] must match
        let select_interior = select_out.clone().slice_dim(1, s![w..n - w]);
        let unfold_interior = unfold_out.clone().slice_dim(1, s![w..n - w]);
        let interior_diff = max_abs_diff(select_interior, unfold_interior);
        assert!(
            interior_diff < 1e-5,
            "Interior tokens should match, diff = {interior_diff}"
        );

        // Boundary tokens must differ (clamp != zero padding, so diff > 0)
        let select_boundary = select_out.slice_dim(1, s![0..w]);
        let unfold_boundary = unfold_out.slice_dim(1, s![0..w]);
        let boundary_diff = max_abs_diff(select_boundary, unfold_boundary);
        assert!(
            boundary_diff > 0.0,
            "Boundary tokens should differ between clamp and zero-pad"
        );
    }

    /// Random input, large window — interior agreement holds at scale.
    #[test]
    fn test_random_input_interior_agreement() {
        let device = device();
        let (b, n, h, dk, w) = (3, 16, 4, 8, 3);

        // Use a fixed seed-like pattern for reproducibility without a seeded RNG
        let data: Vec<f32> = (0..(b * n * h * dk) as i32)
            .map(|i| ((i as f32 * 1.6180339) % 7.0) - 3.5) // pseudo-random spread
            .collect();
        let x =
            Tensor::<B, 4>::from_data(TensorData::new(data, Shape::new([b, n, h, dk])), &device);

        let win_idx = make_window_indices(n, w, &device);
        let select_out = local_window_select(x.clone(), win_idx, w);
        let unfold_out = local_window_unfold(x, w);

        let select_interior = select_out.slice_dim(1, s![w..n - w]);
        let unfold_interior = unfold_out.slice_dim(1, s![w..n - w]);

        let diff = max_abs_diff(select_interior, unfold_interior);
        assert!(diff < 1e-5, "Random input interior diff = {diff}");
    }

    /// Window of 1 (w=0, bw=1) — trivially each token attends only itself.
    /// Both approaches must return the input unchanged (modulo reshape).
    #[test]
    fn test_window_size_one() {
        let device = device();
        let (b, n, h, dk, w) = (2, 5, 2, 4, 0);

        let data: Vec<f32> = (0..(b * n * h * dk) as i32).map(|i| i as f32).collect();
        let x =
            Tensor::<B, 4>::from_data(TensorData::new(data, Shape::new([b, n, h, dk])), &device);

        let win_idx = make_window_indices(n, w, &device);
        let select_out = local_window_select(x.clone(), win_idx, w); // [B,N,H,dk,1]
        let unfold_out = local_window_unfold(x.clone(), w); // [B,N,H,dk,1]

        // Both should equal x reshaped to [B,N,H,dk,1]
        let x_expanded = x.unsqueeze_dim(4);
        let diff_select = max_abs_diff(select_out, x_expanded.clone());
        let diff_unfold = max_abs_diff(unfold_out, x_expanded);

        assert!(
            diff_select < 1e-5,
            "w=0 select should equal input, diff={diff_select}"
        );
        assert!(
            diff_unfold < 1e-5,
            "w=0 unfold should equal input, diff={diff_unfold}"
        );
    }

    /// Output shape is correct for various configurations.
    #[test]
    fn test_output_shapes() {
        let device = device();

        for (b, n, h, dk, w) in [(1, 4, 1, 2, 1), (2, 8, 4, 16, 2), (1, 16, 8, 32, 4)] {
            let bw = 2 * w + 1;
            let data = vec![0.0f32; b * n * h * dk];
            let x = Tensor::<B, 4>::from_data(
                TensorData::new(data, Shape::new([b, n, h, dk])),
                &device,
            );

            let win_idx = make_window_indices(n, w, &device);
            let select_out = local_window_select(x.clone(), win_idx, w);
            let unfold_out = local_window_unfold(x, w);

            assert_eq!(
                select_out.dims(),
                [b, n, h, dk, bw],
                "select shape wrong for b={b} n={n} h={h} dk={dk} w={w}"
            );
            assert_eq!(
                unfold_out.dims(),
                [b, n, h, dk, bw],
                "unfold shape wrong for b={b} n={n} h={h} dk={dk} w={w}"
            );
        }
    }

    /// Specific values — manually verify a tiny case end-to-end.
    /// x = [[1,2],[3,4],[5,6]] (n=3, dk=2, b=1, h=1, w=1)
    /// For token n=1 (middle), window should be [token0, token1, token2]
    /// i.e. along dk dim: [[1,2],[3,4],[5,6]] → last dim of output at n=1
    #[test]
    fn test_specific_values() {
        let device = device();
        let (b, n, h, dk, w) = (1, 3, 1, 2, 1);
        let bw = 2 * w + 1;

        // x[0, 0, 0, :] = [1, 2]
        // x[0, 1, 0, :] = [3, 4]
        // x[0, 2, 0, :] = [5, 6]
        let x = Tensor::<B, 4>::from_data(
            TensorData::new(vec![1.0f32, 2., 3., 4., 5., 6.], Shape::new([b, n, h, dk])),
            &device,
        );

        let win_idx = make_window_indices(n, w, &device);
        let select_out = local_window_select(x.clone(), win_idx, w); // [1,3,1,2,3]
        let unfold_out = local_window_unfold(x, w);

        // At n=1 (middle token), both approaches see tokens [0,1,2] — no boundary effect
        // select_out[0, 1, 0, :, :] should be [[1,3,5],[2,4,6]]
        // i.e. dk=0 row: [tok0_dk0, tok1_dk0, tok2_dk0] = [1, 3, 5]
        //      dk=1 row: [tok0_dk1, tok1_dk1, tok2_dk1] = [2, 4, 6]
        let select_mid = select_out.slice_dim(1, s![1..2]).reshape([dk, bw]);
        let unfold_mid = unfold_out.slice_dim(1, s![1..2]).reshape([dk, bw]);

        let expected = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0f32, 3., 5., 2., 4., 6.], Shape::new([dk, bw])),
            &device,
        );

        let diff_select = (select_mid - expected.clone()).abs().max().into_scalar();
        let diff_unfold = (unfold_mid - expected).abs().max().into_scalar();

        assert!(
            diff_select < 1e-5,
            "select middle token wrong: diff={diff_select}"
        );
        assert!(
            diff_unfold < 1e-5,
            "unfold middle token wrong: diff={diff_unfold}"
        );
    }
}
