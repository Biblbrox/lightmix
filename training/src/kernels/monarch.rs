use cubecl::prelude::*;

#[cube(launch)]
pub fn monarch_fused_kernel<F: Float>(
    x: &Tensor<F>,
    left: &Tensor<F>,
    right: &Tensor<F>,
    out: &mut Tensor<F>,
    #[comptime] a: usize,
    #[comptime] b: usize,
    #[comptime] d: usize,
    #[comptime] tile_bn: usize,
) {
    let mut smem_left = SharedMemory::<F>::new(a * a * b);
    let mut smem_right = SharedMemory::<F>::new(a * d * a);

    let tid = UNIT_POS_X as usize;
    let cube_dim = CUBE_DIM_X as usize;
    let bn_start = CUBE_POS_X as usize * tile_bn;
    let bn_total = x.shape(0);

    let left_size = a * a * b;
    let right_size = a * d * a;

    let mut i = tid;
    while i < left_size {
        smem_left[i] = left[i];
        i += cube_dim;
    }

    let mut j = tid;
    while j < right_size {
        smem_right[j] = right[j];
        j += cube_dim;
    }

    //cubecl::prelude::sync_units();
    sync_cube();

    if tid >= a * d {
        terminate!();
    }

    let a1 = tid / d;
    let d_ = tid % d;

    for bn_local in 0..tile_bn {
        let bn = bn_start + bn_local;
        if bn >= bn_total {
            terminate!();
        }

        let mut acc = F::new(0.0);

        for a0 in 0..a {
            let r = smem_right[a1 * d * a + d_ * a + a0];
            let mut inter = F::new(0.0);
            for b_ in 0..b {
                inter += smem_left[a0 * a * b + a1 * b + b_] * x[bn * (a * b) + a0 * b + b_];
            }
            acc += r * inter;
        }

        out[bn * (a * d) + tid] = acc;
    }
}
