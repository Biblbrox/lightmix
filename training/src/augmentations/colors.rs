use std::marker::PhantomData;

use burn::{
    Tensor,
    prelude::Backend,
    tensor::{Shape, TensorPrimitive, ops::ConvOptions},
};
use rand::RngExt;

use crate::augmentations::Augmentation;

pub struct ColorJitter<B: Backend> {
    brightness: f32,
    contrast: f32,
    saturation: f32,
    ph: PhantomData<B>,
}

pub struct RandomGrayscale<B: Backend> {
    p: f64,
    ph: PhantomData<B>,
}

impl<B: Backend> ColorJitter<B> {
    pub fn new(brightness: f32, contrast: f32, saturation: f32) -> ColorJitter<B> {
        ColorJitter {
            brightness,
            contrast,
            saturation,
            ph: PhantomData,
        }
    }
}

impl<B: Backend> RandomGrayscale<B> {
    pub fn new(p: f64) -> RandomGrayscale<B> {
        RandomGrayscale { p, ph: PhantomData }
    }
}

impl<B: Backend> Augmentation<B> for ColorJitter<B> {
    // Input shape: [B, C, H, W]
    fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let shape = input.shape();

        let mut rng = rand::rng();
        let br = [1.0 + rng.random_range(-self.brightness..=self.brightness); 1];
        let ctr = [1.0 + rng.random_range(-self.contrast..=self.contrast); 1];
        let st = [1.0 + rng.random_range(-self.saturation..=self.saturation); 1];

        let brightness = Tensor::<B, 1>::from_floats(br, &input.device()).reshape([1, 1, 1, 1]);
        let contrast = Tensor::<B, 1>::from_floats(ctr, &input.device()).reshape([1, 1, 1, 1]);
        let saturation = Tensor::<B, 1>::from_floats(st, &input.device()).reshape([1, 1, 1, 1]);

        let brightness = brightness
            .clone()
            .expand(Shape::new([shape[0], 1, shape[2], shape[3]]));
        let contrast = contrast
            .clone()
            .expand(Shape::new([shape[0], 1, shape[2], shape[3]]));
        let saturation = saturation
            .clone()
            .expand(Shape::new([shape[0], 1, shape[2], shape[3]]));

        // Adjust brightness
        let mut res = (input.clone() * brightness).clamp(0.0, 1.0);

        // Adjust contrast
        let mean = input.clone().mean().into_scalar();
        res = ((res - mean) * contrast.clone() + mean).clamp(0.0, 1.0);

        // Adjust saturation
        let r = res
            .clone()
            .slice([0..shape[0], 0..1, 0..shape[2], 0..shape[3]]);
        let g = res
            .clone()
            .slice([0..shape[0], 1..2, 0..shape[2], 0..shape[3]]);
        let b = res
            .clone()
            .slice([0..shape[0], 2..3, 0..shape[2], 0..shape[3]]);

        let gray = r.clone() * 0.2989 + g.clone() * 0.5870 + b.clone() * 0.1140;

        let r_new = ((r - gray.clone()) * saturation.clone() + gray.clone()).clamp(0.0, 1.0);
        let g_new = ((g - gray.clone()) * saturation.clone() + gray.clone()).clamp(0.0, 1.0);
        let b_new = ((b - gray.clone()) * saturation.clone() + gray.clone()).clamp(0.0, 1.0);

        Tensor::cat(vec![r_new, g_new, b_new], 1)
    }
}

impl<B: Backend> Augmentation<B> for RandomGrayscale<B> {
    // Input shape: [B, C, H, W]
    fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let shape = input.shape();

        let mut rng = rand::rng();
        if rng.random_bool(self.p) {
            let r = input
                .clone()
                .slice([0..shape[0], 0..1, 0..shape[2], 0..shape[3]]);
            let g = input
                .clone()
                .slice([0..shape[0], 1..2, 0..shape[2], 0..shape[3]]);
            let b = input
                .clone()
                .slice([0..shape[0], 2..3, 0..shape[2], 0..shape[3]]);

            let gray = r.clone() * 0.2989 + g.clone() * 0.5870 + b.clone() * 0.1140;
            return Tensor::cat(vec![gray.clone(), gray.clone(), gray], 1);
        }
        input
    }
}

/// Random Erasing (Zhong et al. 2020)
pub struct RandomErasing<B: Backend> {
    p: f64,
    min_scale: f64,
    max_scale: f64,
    min_ratio: f64,
    max_ratio: f64,
    fill_value: f32,
    max_attempts: usize,
    ph: PhantomData<B>,
}

impl<B: Backend> RandomErasing<B> {
    pub fn new() -> Self {
        Self {
            p: 0.5,
            min_scale: 0.02,
            max_scale: 0.33,
            min_ratio: 0.3,
            max_ratio: 3.3,
            fill_value: 0.0,
            max_attempts: 10,
            ph: PhantomData,
        }
    }

    pub fn with_p(mut self, p: f64) -> Self {
        self.p = p;
        self
    }

    pub fn with_scale(mut self, min: f64, max: f64) -> Self {
        self.min_scale = min;
        self.max_scale = max;
        self
    }

    pub fn with_ratio(mut self, min: f64, max: f64) -> Self {
        self.min_ratio = min;
        self.max_ratio = max;
        self
    }

    pub fn with_fill(mut self, value: f32) -> Self {
        self.fill_value = value;
        self
    }
}

impl<B: Backend> Default for RandomErasing<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Augmentation<B> for RandomErasing<B> {
    // input: [B, C, H, W]
    fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut rng = rand::rng();
        if rng.random::<f64>() > self.p {
            return input;
        }

        let [b, c, h, w] = [
            input.dims()[0],
            input.dims()[1],
            input.dims()[2],
            input.dims()[3],
        ];
        let area = (h * w) as f64;
        let device = input.device();

        // Find a valid erase rectangle
        let region = (0..self.max_attempts).find_map(|_| {
            let scale = rng.random_range(self.min_scale..self.max_scale);
            let ratio = rng.random_range(self.min_ratio..self.max_ratio);
            let erase_area = area * scale;

            let eh = ((erase_area / ratio).sqrt() as usize).min(h);
            let ew = ((erase_area * ratio).sqrt() as usize).min(w);

            if eh == 0 || ew == 0 || eh >= h || ew >= w {
                return None;
            }
            let top = rng.random_range(0..h - eh);
            let left = rng.random_range(0..w - ew);
            Some((top, left, eh, ew))
        });

        let (top, left, eh, ew) = match region {
            Some(r) => r,
            None => return input,
        };

        let mut mask_data = vec![1f32; b * c * h * w];
        for bi in 0..b {
            for ci in 0..c {
                for hi in top..top + eh {
                    for wi in left..left + ew {
                        let idx = bi * (c * h * w) + ci * (h * w) + hi * w + wi;
                        mask_data[idx] = 0.0;
                    }
                }
            }
        }

        let mask = Tensor::<B, 1>::from_floats(mask_data.as_slice(), &device).reshape([b, c, h, w]);

        let fill = Tensor::<B, 4>::full([b, c, h, w], self.fill_value as f64, &device);

        input * mask.clone() + fill * (mask.neg() + 1.0)
    }
}

pub struct GaussianBlur<B: Backend> {
    kernel_size: usize, // must be odd
    min_sigma: f64,
    max_sigma: f64,
    p: f64,
    device: B::Device,
    ph: PhantomData<B>,
}

impl<B: Backend> GaussianBlur<B> {
    pub fn new(kernel_size: usize, device: &B::Device) -> Self {
        assert!(kernel_size % 2 == 1, "kernel_size must be odd");
        Self {
            kernel_size,
            min_sigma: 0.1,
            max_sigma: 2.0,
            p: 0.5,
            device: device.clone(),
            ph: PhantomData,
        }
    }

    pub fn with_sigma(mut self, min: f64, max: f64) -> Self {
        self.min_sigma = min;
        self.max_sigma = max;
        self
    }

    pub fn with_p(mut self, p: f64) -> Self {
        self.p = p;
        self
    }

    fn make_kernel(&self, channels: usize, sigma: f64) -> Tensor<B, 4> {
        let k = self.kernel_size as i32;
        let half = k / 2;
        let mut data = vec![0f32; (k * k) as usize];
        let s2 = (sigma * sigma) as f32;

        let mut sum = 0f32;
        for ky in 0..k {
            for kx in 0..k {
                let dy = (ky - half) as f32;
                let dx = (kx - half) as f32;
                let v = (-(dx * dx + dy * dy) / (2.0 * s2)).exp();
                data[(ky * k + kx) as usize] = v;
                sum += v;
            }
        }
        for v in data.iter_mut() {
            *v /= sum;
        }

        let base = Tensor::<B, 1>::from_floats(data.as_slice(), &self.device).reshape([
            1,
            1,
            self.kernel_size,
            self.kernel_size,
        ]);

        // repeat across channel dimension
        base.repeat(&[channels, 1, 1, 1])
    }
}

impl<B: Backend> Augmentation<B> for GaussianBlur<B> {
    // input: [B, C, H, W]
    fn execute(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut rng = rand::rng();
        if rng.random::<f64>() > self.p {
            return input;
        }

        let sigma = rng.random_range(self.min_sigma..self.max_sigma);
        let [b, c, h, w] = [
            input.dims()[0],
            input.dims()[1],
            input.dims()[2],
            input.dims()[3],
        ];

        let kernel = self.make_kernel(c, sigma); // [C, 1, k, k]
        let pad = self.kernel_size / 2;

        // Depthwise convolution: groups = C keeps each channel independent
        // ConvOptions::new(stride, padding, dilation, groups)
        let options = ConvOptions::new([1, 1], [pad, pad], [1, 1], c);

        // Low-level backend op — avoids needing a Module
        Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(B::conv2d(
            input.into_primitive().tensor(),
            kernel.into_primitive().tensor(),
            None,
            options,
        )))
    }
}

#[cfg(test)]
mod tests {}
