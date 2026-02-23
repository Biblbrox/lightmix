use cubecl::{cube, prelude::*};

/// Hadamard transform kernel. For now, it works with one dimensional tensors only
#[cube(launch)]
pub fn hadamard_transform_kernel<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    let n = input.shape(output.rank() - 1);

    if n == 0 {
        terminate!();
    }

    //if !n.is_power_of_two() {
    //    return Err("Input length must be a power of 2");
    //}

    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..(i + h) {
                let x = input[j];
                let y = input[j + h];
                output[j] = x + y;
                output[j + h] = x - y;
            }
            // CubeCL doesn't understand step_by. That's why this ugly approach is used
            i = i + h * 2;
        }
        h *= 2;
    }
}
