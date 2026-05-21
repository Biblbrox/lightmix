use burn::{
    Tensor,
    tensor::{activation::softmax, backend::Backend},
};

pub mod cascadedattention;
pub mod learnedmixer;
pub mod staticmixer;
pub mod stochasticmixer;
pub mod stochasticwindowmixer;

pub fn sinkhorn<B: Backend>(s: Tensor<B, 4>, temp: f32) -> Tensor<B, 4> {
    let s = softmax(s / temp, 3); // rows
    softmax(s.swap_dims(2, 3), 3).swap_dims(2, 3) // cols
}
