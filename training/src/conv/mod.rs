use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::backend::Backend,
};

#[derive(Module, Debug)]
pub struct PointwiseConvBn<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
}

#[derive(Config, Debug)]
pub struct PointwiseConvBnConfig {
    pub in_channels: usize,
    pub out_channels: usize,
}

impl PointwiseConvBnConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PointwiseConvBn<B> {
        PointwiseConvBn {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [1, 1])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            bn: BatchNormConfig::new(self.out_channels).init(device),
        }
    }
}

impl<B: Backend> PointwiseConvBn<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.bn.forward(self.conv.forward(x))
    }
}

#[derive(Module, Debug)]
pub struct DepthwiseConvBnAct<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
    act: Relu,
}

#[derive(Config, Debug)]
pub struct DepthwiseConvBnActConfig {
    pub channels: usize,
    #[config(default = 3)]
    pub kernel_size: usize,
}

impl DepthwiseConvBnActConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DepthwiseConvBnAct<B> {
        DepthwiseConvBnAct {
            conv: Conv2dConfig::new(
                [self.channels, self.channels],
                [self.kernel_size, self.kernel_size],
            )
            .with_groups(self.channels)
            .with_padding(PaddingConfig2d::Same)
            .init(device),
            bn: BatchNormConfig::new(self.channels).init(device),
            act: Relu::new(),
        }
    }
}

impl<B: Backend> DepthwiseConvBnAct<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        self.act.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct ConvBNAct<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
    activation: Relu,
}

impl<B: Backend> ConvBNAct<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        self.activation.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct ConvBNActConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    #[config(default = 1)]
    pub stride: usize,
    #[config(default = 1)]
    pub groups: usize,
}

impl ConvBNActConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvBNAct<B> {
        ConvBNAct {
            conv: Conv2dConfig::new(
                [self.in_channels, self.out_channels],
                [self.kernel_size, self.kernel_size],
            )
            .with_stride([self.stride, self.stride])
            .with_groups(self.groups)
            .with_padding(PaddingConfig2d::Same)
            .init(device),

            bn: BatchNormConfig::new(self.out_channels).init(device),

            activation: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct MBConv<B: Backend> {
    expand: ConvBNAct<B>,
    depthwise: ConvBNAct<B>,
    project: ConvBNAct<B>,
    use_residual: bool,
}

#[derive(Config, Debug)]
pub struct MBConvConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    #[config(default = 4)]
    pub expand_ratio: usize,
    #[config(default = 1)]
    pub stride: usize,
}

impl MBConvConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MBConv<B> {
        let hidden_dim = self.in_channels * self.expand_ratio;

        MBConv {
            expand: ConvBNActConfig::new(self.in_channels, hidden_dim, 1).init(device),

            depthwise: ConvBNActConfig::new(hidden_dim, hidden_dim, 3)
                .with_stride(self.stride)
                .with_groups(hidden_dim)
                .init(device),

            project: ConvBNActConfig::new(hidden_dim, self.out_channels, 1).init(device),

            use_residual: self.stride == 1 && self.in_channels == self.out_channels,
        }
    }
}

impl<B: Backend> MBConv<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = x.clone();

        let x = self.expand.forward(x);
        let x = self.depthwise.forward(x);
        let x = self.project.forward(x);

        if self.use_residual { x + residual } else { x }
    }
}
