use burn::{
    module::{Module, Parameter},
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::Distribution,
};

#[derive(Module, Debug)]
pub struct MHPermutMix<B: Backend> {
    signs: Vec<Vec<Tensor<B, 1>>>,
    perms: Vec<Vec<Tensor<B, 1, Int>>>,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct MHPermutMixConfig {
    embed_dim: usize,
    seq_length: usize,
    num_heads: usize,
    out_channels: usize,
    num_encoders: usize,
}

impl<B: Backend> MHPermutMix<B> {
    pub fn forward(&self, x: Tensor<B, 3>, encoder_num: usize) -> Tensor<B, 3> {
        let shape = x.shape(); // [B, N, E]
        let x = x.reshape([shape[0], shape[1] * shape[2]]); // [B, N * E]
        assert_eq!(x.shape(), Shape::new([shape[0], shape[1] * shape[2]]));
        let mut fused_permuted = Vec::<Tensor<B, 3>>::new();
        for i in 0..self.signs[0].len() {
            let transform = self.perms[encoder_num][i].clone(); // [N * E]
            assert_eq!(transform.shape(), Shape::new([shape[1] * shape[2]]));
            let signs = self.signs[encoder_num][i].clone();
            assert_eq!(signs.shape(), Shape::new([shape[1] * shape[2]]));
            let permuted = x.clone().select(1, transform) * signs.unsqueeze();
            let permuted = permuted.reshape(shape.clone());
            fused_permuted.push(permuted);
        }
        let x = Tensor::<B, 3>::cat(fused_permuted, 2);
        assert_eq!(
            x.shape(),
            Shape::new([shape[0], shape[1], shape[2] * self.signs[0].len()])
        );
        let x = self.linear.forward(x);
        return x;
    }
}

impl MHPermutMixConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MHPermutMix<B> {
        let distribution = Distribution::Uniform(-1.0, 1.0);
        let d = self.embed_dim * self.seq_length;
        let mut perms_per_encoder = Vec::<Vec<Tensor<B, 1, Int>>>::new();
        let mut sign_per_encoder = Vec::<Vec<Tensor<B, 1>>>::new();
        (0..self.num_encoders).for_each(|_| {
            let mut perms_per_head = Vec::<Tensor<B, 1, Int>>::new();
            let mut sign_per_head = Vec::<Tensor<B, 1>>::new();
            (0..self.num_heads).for_each(|_| {
                let rand = Tensor::<B, 1>::random(
                    Shape::new([d]),
                    Distribution::Uniform(0.0, 1.0),
                    device,
                );

                perms_per_head.push(rand.argsort(0).set_require_grad(false));
                sign_per_head.push(
                    Tensor::<B, 1>::random(
                        Shape::new([self.embed_dim * self.seq_length]),
                        distribution,
                        device,
                    )
                    .sign()
                    .set_require_grad(false),
                )
            });
            perms_per_encoder.push(perms_per_head);
            sign_per_encoder.push(sign_per_head);
        });

        MHPermutMix {
            signs: sign_per_encoder,
            perms: perms_per_encoder,
            linear: LinearConfig::new(self.embed_dim * self.num_heads, self.out_channels)
                .init(device),
        }
    }
}
