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
pub struct Patcher<B: Backend> {
    conv: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct PatcherConfig {
    in_channels: usize,
    embed_dim: usize,
    patch_size: usize,
}

#[derive(Module, Debug)]
pub struct PatchEmbedding<B: Backend> {
    patcher: Patcher<B>,
    position_embeddings: Param<Tensor<B, 3>>,
    cls: Option<Param<Tensor<B, 3>>>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct PatchEmbeddingConfig {
    in_channels: usize,
    embed_dim: usize,
    patch_size: usize,
    image_size: usize,
    dropout: f64,
    seq_length: usize,
    use_cls: bool,
}

impl<B: Backend> Patcher<B> {
    // # Shapes
    // - Images: [batch_size, num_channels, height, width]
    // - Output: [batch_size, num_channels, num_patches, embed_dim]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 3> {
        let x = self.conv.forward(images); // [batch_size, embed_dim, row_patch_num, row_patch_num]
        let x = x.flatten(2, 3); // [batch_suze, embed_dim, total_patch_num]
        x.transpose() // [batch_size, total_patch_num, embed_dim]
    }
}

impl PatcherConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Patcher<B> {
        Patcher {
            conv: Conv2dConfig::new(
                [self.in_channels, self.embed_dim],
                [self.patch_size, self.patch_size],
            )
            .with_stride([self.patch_size, self.patch_size])
            .init(device),
        }
    }
}

impl<B: Backend> PatchEmbedding<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 3> {
        let patches = self.patcher.forward(images); // [batch_size, total_patch_dim, embed_dim]
        let mut x = self.position_embeddings.val() + patches;
        if !self.cls.is_none() {
            let [b, _, _] = x.dims();
            x = Tensor::cat(vec![self.cls.clone().unwrap().val().repeat_dim(0, b), x], 1);
        }
        self.dropout.forward(x)
    }
}

impl PatchEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PatchEmbedding<B> {
        let distribution = Distribution::Normal(0.0, 1.0);
        let cls = match self.use_cls {
            true => Some(Param::from_tensor(Tensor::<B, 3>::random(
                [1, 1, self.embed_dim],
                distribution,
                device,
            ))),
            false => None,
        };
        PatchEmbedding {
            patcher: PatcherConfig::new(self.in_channels, self.embed_dim, self.patch_size)
                .init(device),

            position_embeddings: Param::<Tensor<B, 3>>::from_tensor(Tensor::<B, 3>::random(
                Shape::new([1, self.seq_length, self.embed_dim]),
                distribution,
                device,
            ))
            .set_require_grad(true),
            dropout: DropoutConfig::new(self.dropout).init(),
            cls,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Shape;

    use crate::{models::fast_vit::FastViTConfig, tokenization::vit::PatcherConfig};

    use super::*;

    const IN_CHANNELS: usize = 3;
    const PATCH_SIZE: usize = 4;
    const IMG_SIZE: usize = 32;
    const NUM_PATCHES: usize = (IMG_SIZE / PATCH_SIZE).pow(2); // 64
    const EMBED_DIM: usize = PATCH_SIZE.pow(2) * 1;
    const NUM_HEADS: usize = 8;
    const NUM_ENCODERS: usize = 4;
    const NUM_CLASSES: usize = 10;
    const BATCH_SIZE: usize = 10;
    const NUM_CHANNELS: usize = 1;
    const HIDDEN_DIM: usize = 64;
    const DROPOUT: f64 = 0.1;
    const SINKHORNE_TEMPERATURE: f32 = 0.05;

    #[test]
    fn test_patcher() {
        type B = burn::backend::cuda::Cuda;
        let device = burn::backend::cuda::CudaDevice::default();

        // Create test image
        let test_image = Tensor::<B, 4>::zeros(
            Shape::new([BATCH_SIZE, NUM_CHANNELS, IMG_SIZE, IMG_SIZE]),
            &device,
        );

        // Create pather
        let patcher = PatcherConfig::new(IN_CHANNELS, EMBED_DIM, PATCH_SIZE).init(&device);
        let patched_image = patcher.forward(test_image);

        assert_eq!(
            patched_image.shape(),
            Shape::new([BATCH_SIZE, NUM_PATCHES, EMBED_DIM])
        );
    }

    #[test]
    fn test_patch_embedding() {
        type B = burn::backend::cuda::Cuda;
        let device = burn::backend::cuda::CudaDevice::default();

        // Create test image
        let test_image = Tensor::<B, 4>::zeros(
            Shape::new([BATCH_SIZE, NUM_CHANNELS, IMG_SIZE, IMG_SIZE]),
            &device,
        );

        let model = PatchEmbeddingConfig::new(
            IN_CHANNELS,
            EMBED_DIM,
            PATCH_SIZE,
            IMG_SIZE,
            DROPOUT,
            NUM_PATCHES,
            true,
        )
        .init(&device);
        let vit_input = model.forward(test_image);
        assert_eq!(
            vit_input.shape(),
            Shape::new([BATCH_SIZE, NUM_PATCHES + 1, EMBED_DIM])
        );
    }

    #[test]
    fn test_vit() {
        type B = burn::backend::cuda::Cuda;
        let device = burn::backend::cuda::CudaDevice::default();

        // Create test image
        let test_image = Tensor::<B, 4>::zeros(
            Shape::new([BATCH_SIZE, NUM_CHANNELS, IMG_SIZE, IMG_SIZE]),
            &device,
        );

        let model = FastViTConfig::new(
            IN_CHANNELS,
            EMBED_DIM,
            NUM_HEADS,
            NUM_ENCODERS,
            NUM_CLASSES,
            PATCH_SIZE,
            IMG_SIZE,
            HIDDEN_DIM,
            DROPOUT,
            SINKHORNE_TEMPERATURE,
        )
        .init(&device);
        let vit_output = model.forward(test_image);
        assert_eq!(vit_output.shape(), Shape::new([BATCH_SIZE, NUM_CLASSES]));
    }
}
