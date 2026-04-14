use burn::backend::Autodiff;

//pub mod kernels;
pub mod mixing;
pub mod models;
pub mod norm;
pub mod spectre_vit;

pub type GpuBackend = burn::backend::cuda::Cuda;
pub type GpuAutodiffBackend = Autodiff<GpuBackend>;
pub type GpuDevice = burn::backend::cuda::CudaDevice;
pub type CpuBackend = burn::backend::ndarray::NdArray;
pub type CpuAutodiffBackend = Autodiff<CpuBackend>;
pub type CpuDevice = burn::backend::ndarray::NdArrayDevice;
