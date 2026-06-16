//! Rust runtime bindings for the native gfxGRAPH HIP/C++ bridge.
//!
//! This crate intentionally avoids PyO3. It loads `libhipgraph_bridge.so`
//! directly and exposes a small Rust-owned control surface for native runtime
//! health checks, versioning, and telemetry.

use libc::{c_char, c_int, c_void};
use std::env;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HgbVersion {
    pub major: c_int,
    pub minor: c_int,
    pub patch: c_int,
    pub gfx_target: *const c_char,
    pub rocm_version: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProfileSample {
    pub seq: u64,
    pub timestamp_ns: u64,
    pub duration_ns: u64,
    pub value0: u64,
    pub value1: u64,
    pub event: u32,
    pub device_id: u32,
    pub stream_id: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProfileCounters {
    pub written: u64,
    pub dropped: u64,
    pub capacity: u64,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileEventKind {
    Unknown = 0,
    PipelineCreate = 1,
    PipelineUpload = 2,
    PipelineLaunch = 3,
    PipelineUpdate = 4,
    ComposeLaunch = 5,
    ShapeLaunch = 6,
    DeviceClockProbe = 7,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeVersion {
    pub major: i32,
    pub minor: i32,
    pub patch: i32,
    pub gfx_target: String,
    pub rocm_version: String,
}

#[derive(Debug)]
pub enum NativeLoadError {
    Open { path: PathBuf, message: String },
    Symbol { name: &'static str, message: String },
    NoCandidate { candidates: Vec<PathBuf> },
    InvalidSymbolName(&'static str),
}

impl fmt::Display for NativeLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open { path, message } => {
                write!(f, "failed to open {}: {message}", path.display())
            }
            Self::Symbol { name, message } => write!(f, "failed to load symbol {name}: {message}"),
            Self::NoCandidate { candidates } => {
                write!(f, "no native bridge library candidate exists:")?;
                for candidate in candidates {
                    write!(f, " {}", candidate.display())?;
                }
                Ok(())
            }
            Self::InvalidSymbolName(name) => write!(f, "invalid symbol name: {name}"),
        }
    }
}

impl Error for NativeLoadError {}

#[derive(Debug)]
pub enum NativeRuntimeError {
    HipError(i32),
    NullHandle(&'static str),
    LengthMismatch { graphs: usize, deps: usize },
}

impl fmt::Display for NativeRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HipError(code) => write!(f, "HIP runtime returned error code {code}"),
            Self::NullHandle(name) => write!(f, "{name} returned a null native handle"),
            Self::LengthMismatch { graphs, deps } => write!(
                f,
                "composed graph input length mismatch: {graphs} graphs, {deps} deps"
            ),
        }
    }
}

impl Error for NativeRuntimeError {}

type HgbInit = unsafe extern "C" fn() -> c_int;
type HgbShutdown = unsafe extern "C" fn();
type HgbIsInitialized = unsafe extern "C" fn() -> c_int;
type HgbGetVersion = unsafe extern "C" fn() -> HgbVersion;
type HgbProfilerReset = unsafe extern "C" fn();
type HgbProfilerRecord = unsafe extern "C" fn(u32, u64, u64, u64) -> u64;
type HgbProfilerSnapshot =
    unsafe extern "C" fn(*mut ProfileSample, usize, *mut ProfileCounters) -> usize;
type HgbPipelineHandleCreate = unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> c_int;
type HgbPipelineHandleLaunch = unsafe extern "C" fn(*mut c_void) -> c_int;
type HgbPipelineHandleUpdateKernel =
    unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> c_int;
type HgbPipelineHandleUpdateAndLaunch =
    unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> c_int;
type HgbPipelineHandleDestroy = unsafe extern "C" fn(*mut c_void);
type HgbComposedHandleCreate =
    unsafe extern "C" fn(*mut *mut c_void, c_int, *const c_int, *mut *mut c_void) -> c_int;
type HgbComposedHandleLaunch = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type HgbComposedHandleUpdateChild = unsafe extern "C" fn(*mut c_void, c_int, *mut c_void) -> c_int;
type HgbComposedHandleDestroy = unsafe extern "C" fn(*mut c_void);

#[derive(Clone, Copy)]
struct NativeApi {
    init: HgbInit,
    shutdown: HgbShutdown,
    is_initialized: HgbIsInitialized,
    get_version: HgbGetVersion,
    profiler_reset: HgbProfilerReset,
    profiler_record: HgbProfilerRecord,
    profiler_snapshot: HgbProfilerSnapshot,
    pipeline_handle_create: HgbPipelineHandleCreate,
    pipeline_handle_launch: HgbPipelineHandleLaunch,
    pipeline_handle_update_kernel: HgbPipelineHandleUpdateKernel,
    pipeline_handle_update_and_launch: HgbPipelineHandleUpdateAndLaunch,
    pipeline_handle_destroy: HgbPipelineHandleDestroy,
    composed_handle_create: HgbComposedHandleCreate,
    composed_handle_launch: HgbComposedHandleLaunch,
    composed_handle_update_child: HgbComposedHandleUpdateChild,
    composed_handle_destroy: HgbComposedHandleDestroy,
}

pub struct NativeBridge {
    handle: *mut c_void,
    path: PathBuf,
    api: NativeApi,
}

unsafe impl Send for NativeBridge {}
unsafe impl Sync for NativeBridge {}

impl NativeBridge {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, NativeLoadError> {
        let path = path.as_ref().to_path_buf();
        let c_path = CString::new(path.as_os_str().to_string_lossy().as_bytes()).map_err(|_| {
            NativeLoadError::Open {
                path: path.clone(),
                message: "path contains interior NUL".to_string(),
            }
        })?;

        clear_dlerror();
        let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return Err(NativeLoadError::Open {
                path,
                message: dlerror_string(),
            });
        }

        let api = unsafe {
            NativeApi {
                init: load_symbol(handle, "hgb_init")?,
                shutdown: load_symbol(handle, "hgb_shutdown")?,
                is_initialized: load_symbol(handle, "hgb_is_initialized")?,
                get_version: load_symbol(handle, "hgb_get_version")?,
                profiler_reset: load_symbol(handle, "hgb_profiler_reset")?,
                profiler_record: load_symbol(handle, "hgb_profiler_record")?,
                profiler_snapshot: load_symbol(handle, "hgb_profiler_snapshot")?,
                pipeline_handle_create: load_symbol(handle, "hgb_pipeline_handle_create")?,
                pipeline_handle_launch: load_symbol(handle, "hgb_pipeline_handle_launch")?,
                pipeline_handle_update_kernel: load_symbol(
                    handle,
                    "hgb_pipeline_handle_update_kernel",
                )?,
                pipeline_handle_update_and_launch: load_symbol(
                    handle,
                    "hgb_pipeline_handle_update_and_launch",
                )?,
                pipeline_handle_destroy: load_symbol(handle, "hgb_pipeline_handle_destroy")?,
                composed_handle_create: load_symbol(handle, "hgb_composed_handle_create")?,
                composed_handle_launch: load_symbol(handle, "hgb_composed_handle_launch")?,
                composed_handle_update_child: load_symbol(
                    handle,
                    "hgb_composed_handle_update_child",
                )?,
                composed_handle_destroy: load_symbol(handle, "hgb_composed_handle_destroy")?,
            }
        };

        Ok(Self { handle, path, api })
    }

    pub fn open_default(repo_root: impl AsRef<Path>) -> Result<Self, NativeLoadError> {
        let candidates = default_library_candidates(repo_root);
        for candidate in &candidates {
            if candidate.exists() {
                return Self::open(candidate);
            }
        }
        Err(NativeLoadError::NoCandidate { candidates })
    }

    pub fn library_path(&self) -> &Path {
        &self.path
    }

    pub fn init(&self) -> Result<(), NativeRuntimeError> {
        match unsafe { (self.api.init)() } {
            0 => Ok(()),
            code => Err(NativeRuntimeError::HipError(code)),
        }
    }

    pub fn shutdown(&self) {
        unsafe { (self.api.shutdown)() };
    }

    pub fn is_initialized(&self) -> bool {
        unsafe { (self.api.is_initialized)() != 0 }
    }

    pub fn version(&self) -> NativeVersion {
        let raw = unsafe { (self.api.get_version)() };
        NativeVersion {
            major: raw.major,
            minor: raw.minor,
            patch: raw.patch,
            gfx_target: cstr_to_string(raw.gfx_target),
            rocm_version: cstr_to_string(raw.rocm_version),
        }
    }

    pub fn profiler_reset(&self) {
        unsafe { (self.api.profiler_reset)() };
    }

    pub fn profiler_record(
        &self,
        event: ProfileEventKind,
        duration_ns: u64,
        value0: u64,
        value1: u64,
    ) -> u64 {
        unsafe { (self.api.profiler_record)(event as u32, duration_ns, value0, value1) }
    }

    pub fn profiler_snapshot(&self, max_samples: usize) -> (Vec<ProfileSample>, ProfileCounters) {
        let mut counters = ProfileCounters::default();
        let mut samples = vec![ProfileSample::default(); max_samples];
        let copied = unsafe {
            (self.api.profiler_snapshot)(
                samples.as_mut_ptr(),
                samples.len(),
                &mut counters as *mut ProfileCounters,
            )
        };
        samples.truncate(copied);
        (samples, counters)
    }

    pub unsafe fn pipeline_from_raw_graph(
        &self,
        graph: *mut c_void,
    ) -> Result<NativePipeline<'_>, NativeRuntimeError> {
        let mut handle = ptr::null_mut();
        let code = unsafe { (self.api.pipeline_handle_create)(graph, &mut handle) };
        if code != 0 {
            return Err(NativeRuntimeError::HipError(code));
        }
        if handle.is_null() {
            return Err(NativeRuntimeError::NullHandle("hgb_pipeline_handle_create"));
        }
        Ok(NativePipeline {
            bridge: self,
            handle,
        })
    }

    pub unsafe fn composed_from_raw_graphs(
        &self,
        sub_graphs: &mut [*mut c_void],
        deps: &[c_int],
    ) -> Result<NativeComposedGraph<'_>, NativeRuntimeError> {
        if sub_graphs.len() != deps.len() {
            return Err(NativeRuntimeError::LengthMismatch {
                graphs: sub_graphs.len(),
                deps: deps.len(),
            });
        }

        let mut handle = ptr::null_mut();
        let code = unsafe {
            (self.api.composed_handle_create)(
                sub_graphs.as_mut_ptr(),
                sub_graphs.len() as c_int,
                deps.as_ptr(),
                &mut handle,
            )
        };
        if code != 0 {
            return Err(NativeRuntimeError::HipError(code));
        }
        if handle.is_null() {
            return Err(NativeRuntimeError::NullHandle("hgb_composed_handle_create"));
        }
        Ok(NativeComposedGraph {
            bridge: self,
            handle,
        })
    }
}

impl Drop for NativeBridge {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                libc::dlclose(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

pub struct NativePipeline<'a> {
    bridge: &'a NativeBridge,
    handle: *mut c_void,
}

impl<'a> NativePipeline<'a> {
    pub fn as_raw(&self) -> *mut c_void {
        self.handle
    }

    pub fn launch(&self) -> Result<(), NativeRuntimeError> {
        match unsafe { (self.bridge.api.pipeline_handle_launch)(self.handle) } {
            0 => Ok(()),
            code => Err(NativeRuntimeError::HipError(code)),
        }
    }

    pub unsafe fn update_kernel(
        &self,
        node: *mut c_void,
        params: *mut c_void,
    ) -> Result<(), NativeRuntimeError> {
        match unsafe { (self.bridge.api.pipeline_handle_update_kernel)(self.handle, node, params) }
        {
            0 => Ok(()),
            code => Err(NativeRuntimeError::HipError(code)),
        }
    }

    pub unsafe fn update_and_launch(
        &self,
        node: *mut c_void,
        params: *mut c_void,
    ) -> Result<(), NativeRuntimeError> {
        match unsafe {
            (self.bridge.api.pipeline_handle_update_and_launch)(self.handle, node, params)
        } {
            0 => Ok(()),
            code => Err(NativeRuntimeError::HipError(code)),
        }
    }
}

impl Drop for NativePipeline<'_> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.bridge.api.pipeline_handle_destroy)(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

pub struct NativeComposedGraph<'a> {
    bridge: &'a NativeBridge,
    handle: *mut c_void,
}

impl<'a> NativeComposedGraph<'a> {
    pub fn as_raw(&self) -> *mut c_void {
        self.handle
    }

    pub unsafe fn launch(&self, stream: *mut c_void) -> Result<(), NativeRuntimeError> {
        match unsafe { (self.bridge.api.composed_handle_launch)(self.handle, stream) } {
            0 => Ok(()),
            code => Err(NativeRuntimeError::HipError(code)),
        }
    }

    pub unsafe fn update_child(
        &self,
        child_index: c_int,
        new_sub_graph: *mut c_void,
    ) -> Result<(), NativeRuntimeError> {
        match unsafe {
            (self.bridge.api.composed_handle_update_child)(self.handle, child_index, new_sub_graph)
        } {
            0 => Ok(()),
            code => Err(NativeRuntimeError::HipError(code)),
        }
    }
}

impl Drop for NativeComposedGraph<'_> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.bridge.api.composed_handle_destroy)(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

pub fn default_library_candidates(repo_root: impl AsRef<Path>) -> Vec<PathBuf> {
    let repo_root = repo_root.as_ref();
    let mut candidates = Vec::new();
    for key in ["GFXGRAPH_NATIVE_LIB", "GFXGRAPH_LIB"] {
        if let Ok(value) = env::var(key) {
            if !value.trim().is_empty() {
                candidates.push(PathBuf::from(value));
            }
        }
    }
    candidates.push(repo_root.join("build/libhipgraph_bridge.so"));
    candidates.push(repo_root.join("build/lib/libhipgraph_bridge.so"));
    candidates.push(PathBuf::from("/usr/local/lib/libhipgraph_bridge.so"));
    candidates
}

unsafe fn load_symbol<T: Copy>(
    handle: *mut c_void,
    name: &'static str,
) -> Result<T, NativeLoadError> {
    let c_name = CString::new(name).map_err(|_| NativeLoadError::InvalidSymbolName(name))?;
    clear_dlerror();
    let symbol = unsafe { libc::dlsym(handle, c_name.as_ptr()) };
    if symbol.is_null() {
        return Err(NativeLoadError::Symbol {
            name,
            message: dlerror_string(),
        });
    }
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&symbol) })
}

fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

fn clear_dlerror() {
    unsafe {
        libc::dlerror();
    }
}

fn dlerror_string() -> String {
    let ptr = unsafe { libc::dlerror() };
    if ptr.is_null() {
        return "unknown dynamic loader error".to_string();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_candidates_include_repo_build_output() {
        let candidates = default_library_candidates("/tmp/gfxGRAPH");
        assert!(candidates
            .iter()
            .any(|path| path.ends_with("build/libhipgraph_bridge.so")));
    }

    #[test]
    fn profile_structs_have_expected_capacity_for_c_abi() {
        assert!(std::mem::size_of::<ProfileSample>() >= 56);
        assert_eq!(std::mem::size_of::<ProfileCounters>(), 24);
    }
}
