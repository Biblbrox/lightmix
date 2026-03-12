use burn::{
    module::{Module, Param},
    nn::{
        Dropout, DropoutConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::Distribution,
};

#[derive(Module, Debug)]
pub struct SpectrePatcher<B: Backend> {
    conv: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct SpectrePatcherConfig {
    in_channels: usize,
    embed_dim: usize,
    patch_size: usize,
}

#[derive(Module, Debug)]
pub struct SpectrePatchEmbedding<B: Backend> {
    patcher: SpectrePatcher<B>,
    cls_token: Param<Tensor<B, 3>>,
    position_embeddings: Param<Tensor<B, 3>>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct SpectrePatchEmbeddingConfig {
    in_channels: usize,
    embed_dim: usize,
    patch_size: usize,
    image_size: usize,
}

impl<B: Backend> SpectrePatcher<B> {
    // # Shapes
    // - Images: [batch_size, num_channels, height, width]
    // - Output: [batch_size, num_channels, num_patches + 1, embed_dim]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 3> {
        let x = self.conv.forward(images); // [batch_size, embed_dim, row_patch_num, row_patch_num]
        let x = x.flatten(2, 3); // [batch_suze, embed_dim, total_patch_num]
        x.swap_dims(1, 2) // [batch_size, total_patch_num, embed_dim]
    }
}

impl SpectrePatcherConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectrePatcher<B> {
        SpectrePatcher {
            conv: Conv2dConfig::new(
                [self.in_channels, self.embed_dim],
                [self.patch_size, self.patch_size],
            )
            .with_stride([self.patch_size, self.patch_size])
            .init(device),
        }
    }
}

impl<B: Backend> SpectrePatchEmbedding<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 3> {
        let patches = self.patcher.forward(images.clone()); // [batch_size, total_patch_dim, embed_dim]
        // Expand cls_token alongside batch dimension. Left other
        // dimensions untouched
        let cls_token_batch = self
            .cls_token
            .val()
            .expand([images.dims()[0] as i32, -1, -1]);
        // Concatenate cls token and image patches
        let x = Tensor::cat(Vec::from([cls_token_batch, patches]), 1);
        let x = self.position_embeddings.val() + x;
        let x = self.dropout.forward(x);
        x // [batch_size, total_patch_dim + 1, embed_dim]
    }
}

impl SpectrePatchEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectrePatchEmbedding<B> {
        let distribution = Distribution::Normal(0.0, 1.0);
        let num_patches = (self.image_size / self.patch_size).pow(2);
        SpectrePatchEmbedding {
            patcher: SpectrePatcherConfig::new(self.in_channels, self.embed_dim, self.patch_size)
                .init(device),
            cls_token: Param::<Tensor<B, 3>>::from_tensor(Tensor::<B, 3>::random(
                Shape::new([1, 1, self.embed_dim]),
                distribution,
                device,
            ))
            .set_require_grad(true),
            position_embeddings: Param::<Tensor<B, 3>>::from_tensor(Tensor::<B, 3>::random(
                Shape::new([1, num_patches + 1, self.embed_dim]),
                distribution,
                device,
            ))
            .set_require_grad(true),
            dropout: DropoutConfig::new(0.001).init(),
        }
    }
}
