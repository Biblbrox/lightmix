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
    let s = s / temp;
    let s = softmax(s, 3); // row normalisation (along last dim)
    softmax(s, 2)
}
