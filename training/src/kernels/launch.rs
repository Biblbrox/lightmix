use burn::tensor::{Shape, TensorMetadata};
use burn_cubecl::{CubeRuntime, FloatElement, ops::numeric::empty_device, tensor::CubeTensor};
use cubecl::prelude::*;

use crate::kernels::monarch::monarch_fused_kernel;

pub fn launch_monarch_fused<R: CubeRuntime, E: FloatElement>(
    client: &ComputeClient<R>,
    x: &CubeTensor<R>,
    left: &CubeTensor<R>,
    right: &CubeTensor<R>,
    a: usize,
    b: usize,
    d: usize,
) -> CubeTensor<R> {
    let bn = x.shape()[0];

    let out = empty_device::<R, E>(client.clone(), x.device.clone(), Shape::new([bn, a * d]));

    let tile_bn: usize = 8;
    let num_cubes = bn.div_ceil(tile_bn) as u32;

    let cube_dim = CubeDim {
        x: (a * d) as u32,
        y: 1,
        z: 1,
    };

    unsafe {
        monarch_fused_kernel::launch::<E, R>(
            client,
            CubeCount::Static(num_cubes, 1, 1),
            cube_dim,
            TensorArg::from_raw_parts(
                x.handle.clone(),
                x.meta.strides.clone(),
                x.meta.shape.clone(),
            ),
            TensorArg::from_raw_parts(
                left.handle.clone(),
                left.meta.strides.clone(),
                left.meta.shape.clone(),
            ),
            TensorArg::from_raw_parts(
                right.handle.clone(),
                right.meta.strides.clone(),
                right.meta.shape.clone(),
            ),
            TensorArg::from_raw_parts(
                out.handle.clone(),
                out.meta.strides.clone(),
                out.meta.shape.clone(),
            ),
            a,
            b,
            d,
            tile_bn,
        );
    }

    out
}
