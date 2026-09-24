//! CUDA backend: device memory and the two embedded fatbins.
//!
//! The driver bindings, the context, the modules, the buffers and the launch
//! marshalling (`vm::{Args, Launch}`) all come from `lightgpu`; what remains
//! here is the f32-facing surface this engine's graph code expects (a buffer
//! counted in ELEMENTS rather than bytes) plus the ASCII-art of its own fatbin.
//! Compiled only with `--features cuda`; `libcuda.so.1` is `dlopen`ed by
//! lightgpu at run time, so the CPU path needs no NVIDIA driver at all.

#![allow(dead_code)]

use lightgpu::ffi::CUdeviceptr;
use lightgpu::vm;

/// The toolkit fatbin produced by build.rs (the generic ops this engine uses).
/// Its `-gencode` set is sm_61 / sm_75 / sm_80 SASS plus compute_80 PTX for
/// forward compatibility.
pub static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmbg_toolkit.fatbin"));

/// This project's own kernel family (`cuda/swin.cu`), loaded as a SECOND
/// module. Separate modules are separate namespaces, so a name in one cannot
/// shadow a name in the other, and a name that is in neither still fails
/// eagerly at startup (see `KERNEL_NAMES`).
pub static SWIN_FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmbg_swin.fatbin"));

/// Device information reported at startup.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub index: i32,
    pub name: String,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub total_mem: usize,
}

pub struct Cuda {
    /// The toolkit module (generic ops). Memoized kernel handles live in here
    /// (lightgpu caches every lookup, which matters because the forward pass
    /// resolves names hundreds of times).
    pub module: vm::Module,
    /// This project's own kernel family, from `cuda/swin.cu`.
    pub swin: vm::Module,
    pub info: DeviceInfo,
}

impl Cuda {
    /// Bring up the driver, device 0, and the embedded fatbin.
    ///
    /// The context comes from `vm::init()`, which binds device 0's PRIMARY
    /// context. An explicit `cuCtxCreate` context would be a second, older
    /// device state and is what the toolkit deliberately avoids.
    pub fn init(verbose: bool) -> Result<Cuda, String> {
        vm::init()?;
        let dev = vm::device()?;
        let module = vm::Module::load(FATBIN)?;
        let swin = vm::Module::load(SWIN_FATBIN)?;
        let c = Cuda {
            module,
            swin,
            info: DeviceInfo {
                index: 0,
                name: dev.name.clone(),
                cc_major: dev.cc_major,
                cc_minor: dev.cc_minor,
                total_mem: 0,
            },
        };
        if verbose {
            eprintln!(
                "cuda: {} cc {}.{} ({} MiB total, fatbins {} + {} bytes)",
                c.info.name,
                c.info.cc_major,
                c.info.cc_minor,
                dev.sm_count * 0, // total_mem is reported below by free_vram
                FATBIN.len(),
                SWIN_FATBIN.len()
            );
        }
        Ok(c)
    }

    pub fn sync(&self) -> Result<(), String> {
        vm::sync()
    }
}

/// A device buffer of f32, counted in ELEMENTS (lightgpu counts bytes; the graph
/// code here thinks in tensor lengths, so the conversion stays in one place).
pub struct DevBuf {
    pub ptr: CUdeviceptr,
    pub len: usize,
    inner: Option<vm::DevBuf>,
}

impl DevBuf {
    /// A zero-element buffer: the graph code uses one as the "nothing to add"
    /// sentinel, and it must not call cuMemAlloc for it.
    pub fn empty() -> DevBuf {
        DevBuf { ptr: 0, len: 0, inner: None }
    }

    pub fn alloc(len: usize) -> Result<DevBuf, String> {
        if len == 0 {
            return Ok(DevBuf::empty());
        }
        let b = vm::DevBuf::alloc(len * std::mem::size_of::<f32>())?;
        Ok(DevBuf { ptr: b.ptr, len, inner: Some(b) })
    }

    pub fn from_host(v: &[f32]) -> Result<DevBuf, String> {
        let b = DevBuf::alloc(v.len())?;
        b.upload(v)?;
        Ok(b)
    }

    pub fn upload(&self, v: &[f32]) -> Result<(), String> {
        if v.len() != self.len {
            return Err(format!("upload size {} != buffer {}", v.len(), self.len));
        }
        match &self.inner {
            Some(b) => b.upload(v),
            None => Ok(()),
        }
    }

    pub fn download(&self, out: &mut [f32]) -> Result<(), String> {
        if out.len() != self.len {
            return Err(format!("download size {} != buffer {}", out.len(), self.len));
        }
        match &self.inner {
            Some(b) => b.download(out),
            None => Ok(()),
        }
    }
}

