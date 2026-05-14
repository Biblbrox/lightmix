/// SpectralPatcher — patch embedding with graph Laplacian positional encodings.
///
/// At init time, the 2D grid of patches is treated as a 4-connected graph.
/// The eigenvectors of its normalized Laplacian are precomputed (via nalgebra)
/// and stored as the positional embeddings. This gives each position a
/// spectrally-aware coordinate that encodes both local neighbourhood and
/// global graph structure — unlike sinusoidal or learned-from-scratch embeddings.
use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    tensor::{Shape, Tensor, TensorData, backend::Backend},
};
use nalgebra::{DMatrix, SymmetricEigen};

// ── Module ────────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct SpectralPatcher<B: Backend> {
    patch_proj: Linear<B>,          // [patch_dim → embed_dim]
    pos_embed: Param<Tensor<B, 2>>, // [N, E] — spectral positional embeddings
    patch_size: usize,
    in_channels: usize,
    grid_h: usize,
    grid_w: usize,
}

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct SpectralPatcherConfig {
    pub image_h: usize,
    pub image_w: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub embed_dim: usize,
    /// If true, spectral embeddings are the initialization but stay learnable.
    /// If false, they are frozen throughout training.
    #[config(default = true)]
    pub learnable_pos: bool,
}

// ── Laplacian eigenvector computation ────────────────────────────────────────

/// Builds a [N, embed_dim] positional embedding matrix from the eigenvectors
/// of the normalized Laplacian of the patch grid graph.
///
/// The grid has 4-connectivity (up/down/left/right). The normalized Laplacian
/// is L_sym = I - D^{-1/2} A D^{-1/2}, whose eigenvectors form a Fourier-like
/// basis over the graph — smooth, global eigenvectors first, oscillatory ones last.
///
/// The first eigenvector (eigenvalue ≈ 0, constant vector) is always skipped.
/// If embed_dim > N-1 (more dims than available eigenvectors), remaining
/// columns are zero-padded.
fn build_spectral_pos_embed(grid_h: usize, grid_w: usize, embed_dim: usize) -> Vec<f32> {
    let n = grid_h * grid_w;

    // ── Adjacency matrix (4-connected grid) ──────────────────────────────────
    let mut adj = DMatrix::<f64>::zeros(n, n);
    for i in 0..grid_h {
        for j in 0..grid_w {
            let idx = i * grid_w + j;
            // Right neighbour
            if j + 1 < grid_w {
                adj[(idx, idx + 1)] = 1.0;
                adj[(idx + 1, idx)] = 1.0;
            }
            // Bottom neighbour
            if i + 1 < grid_h {
                adj[(idx, idx + grid_w)] = 1.0;
                adj[(idx + grid_w, idx)] = 1.0;
            }
        }
    }

    // ── Degree vector ─────────────────────────────────────────────────────────
    // Interior: degree 4 | edges: 3 | corners: 2
    let degrees: Vec<f64> = (0..n).map(|i| adj.row(i).sum()).collect();

    // ── Normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2} ───────────────
    let mut lap = DMatrix::<f64>::identity(n, n);
    for i in 0..n {
        for j in 0..n {
            if adj[(i, j)] > 0.0 {
                // All connected nodes have degree >= 2, so no div-by-zero risk
                lap[(i, j)] = -1.0 / (degrees[i] * degrees[j]).sqrt();
            }
        }
    }

    // ── Symmetric eigendecomposition ─────────────────────────────────────────
    let eigen = SymmetricEigen::new(lap);

    // Sort column indices by eigenvalue ascending
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eigen.eigenvalues[a]
            .partial_cmp(&eigen.eigenvalues[b])
            .unwrap()
    });

    // Skip first trivial eigenvector (eigenvalue ≈ 0), take up to embed_dim
    let selected: Vec<usize> = order.into_iter().skip(1).take(embed_dim).collect();

    // ── Pack into [N, embed_dim] row-major ────────────────────────────────────
    // Columns beyond selected.len() stay 0 (zero-pad when embed_dim > N-1)
    let mut result = vec![0.0_f32; n * embed_dim];
    for (d, &col) in selected.iter().enumerate() {
        for i in 0..n {
            result[i * embed_dim + d] = eigen.eigenvectors[(i, col)] as f32;
        }
    }
    result
}

// ── Init ──────────────────────────────────────────────────────────────────────

impl SpectralPatcherConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectralPatcher<B> {
        assert!(
            self.image_h % self.patch_size == 0 && self.image_w % self.patch_size == 0,
            "Image dimensions ({}, {}) must be divisible by patch_size {}",
            self.image_h,
            self.image_w,
            self.patch_size
        );

        let grid_h = self.image_h / self.patch_size;
        let grid_w = self.image_w / self.patch_size;
        let n = grid_h * grid_w;
        let patch_dim = self.patch_size * self.patch_size * self.in_channels;

        // Precomputed once at init — O(N³) eigen, but N = (H/p)*(W/p) is small
        // e.g. 224/16 × 224/16 = 196 patches → 196×196 eigen, trivially fast
        let pos_data = build_spectral_pos_embed(grid_h, grid_w, self.embed_dim);
        let pos_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(pos_data, Shape::new([n, self.embed_dim])),
            device,
        );

        SpectralPatcher {
            patch_proj: LinearConfig::new(patch_dim, self.embed_dim)
                .with_bias(true)
                .init(device),
            pos_embed: Param::from_tensor(pos_tensor).set_require_grad(self.learnable_pos),
            patch_size: self.patch_size,
            in_channels: self.in_channels,
            grid_h,
            grid_w,
        }
    }
}

// ── Forward ───────────────────────────────────────────────────────────────────

impl<B: Backend> SpectralPatcher<B> {
    /// x: [B, C, H, W] → tokens: [B, N, E]
    ///
    /// Patch extraction via reshape + dim permutation (no unfold, no copy):
    ///
    ///   [B, C, H, W]
    ///   → [B, C, gh, p, gw, p]   reshape  (H = gh*p, W = gw*p, row-major valid)
    ///   → [B, gh, C, p, gw, p]   swap(1,2)
    ///   → [B, gh, gw, p, C, p]   swap(2,4)
    ///   → [B, gh, gw, C, p, p]   swap(3,4)
    ///   → [B, N, patch_dim]      reshape  (N = gh*gw, patch_dim = C*p*p)
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, _c, _h, _w] = x.dims();
        let p = self.patch_size;
        let gh = self.grid_h;
        let gw = self.grid_w;
        let c = self.in_channels;
        let n = gh * gw;
        let patch_dim = p * p * c;

        let tokens = self.patch_proj.forward(
            x.reshape([b, c, gh, p, gw, p]) // [B, C,  gh, p_h, gw, p_w]
                .swap_dims(1, 2) // [B, gh, C,  p_h, gw, p_w]
                .swap_dims(2, 4) // [B, gh, gw, p_h, C,  p_w]
                .swap_dims(3, 4) // [B, gh, gw, C,   p_h, p_w]
                .reshape([b, n, patch_dim]), // [B, N, patch_dim]
        ); // [B, N, E]

        // Broadcast [N, E] → [B, N, E]
        tokens + self.pos_embed.val().unsqueeze_dim(0)
    }
}
