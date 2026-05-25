use burn::backend::Autodiff;

pub mod utils;

pub type GpuBackend = burn::backend::cuda::Cuda;
pub type GpuAutodiffBackend = Autodiff<GpuBackend>;
pub type GpuDevice = burn::backend::cuda::CudaDevice;

//pub type CpuBackend = burn::backend::flex::Flex;
//pub type CpuAutodiffBackend = Autodiff<CpuBackend>;
//pub type CpuDevice = burn::backend::flex::FlexDevice;
pub type CpuBackend = burn::backend::cpu::Cpu;
pub type CpuAutodiffBackend = Autodiff<CpuBackend>;
pub type CpuDevice = burn::backend::cpu::CpuDevice;
