#![cfg(target_os = "linux")]

use std::env;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::process::Command;

use ort::execution_providers::TensorRTExecutionProvider;
use ort::session::Session;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

const RTLD_NOW: i32 = 2;
const RTLD_GLOBAL: i32 = 0x100;

fn provider_candidates(lib_name: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dir) = env::var_os("ORT_DYLIB_PATH") {
        candidates.push(PathBuf::from(dir).join(lib_name));
    }
    if let Some(dir) = env::var_os("ORT_LIB_LOCATION") {
        candidates.push(PathBuf::from(dir).join(lib_name));
    }
    if let Some(home) = env::var_os("HOME") {
        let base = PathBuf::from(home).join(".cache/ort.pyke.io/dfbin");
        if let Ok(triples) = std::fs::read_dir(base) {
            for triple in triples.flatten() {
                if let Ok(hashes) = std::fs::read_dir(triple.path()) {
                    for hash in hashes.flatten() {
                        let p = hash.path().join(lib_name);
                        if p.is_file() {
                            candidates.push(p);
                        }
                    }
                }
            }
        }
    }
    if let Ok(exe) = env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join(lib_name));
        candidates.push(dir.join("deps").join(lib_name));
    }
    candidates
}

fn dlopen_path(path: &Path, flags: i32) -> Result<(), String> {
    let cpath =
        CString::new(path.to_string_lossy().as_bytes()).map_err(|_| "invalid path".to_string())?;
    // SAFETY: `dlopen` expects a valid, NUL-terminated path.
    let handle = unsafe { dlopen(cpath.as_ptr(), flags) };
    if handle.is_null() {
        // SAFETY: `dlerror` returns a pointer to thread-local error string or null.
        let err = unsafe {
            let p = dlerror();
            if p.is_null() {
                "unknown".to_string()
            } else {
                CStr::from_ptr(p).to_string_lossy().to_string()
            }
        };
        Err(format!("dlopen {} failed: {err}", path.display()))
    } else {
        Ok(())
    }
}

#[test]
#[ignore = "requires ORT TensorRT provider libs present on host"]
fn providers_shared_then_tensorrt_dlopen_smoke() {
    let shared = provider_candidates("libonnxruntime_providers_shared.so")
        .into_iter()
        .next()
        .expect("providers_shared not found");
    let trt = provider_candidates("libonnxruntime_providers_tensorrt.so")
        .into_iter()
        .next()
        .expect("providers_tensorrt not found");
    let out = Command::new("nm")
        .arg("-D")
        .arg("--defined-only")
        .arg(&shared)
        .output()
        .expect("run nm");
    assert!(out.status.success(), "nm failed for {}", shared.display());
    let nm_text = String::from_utf8_lossy(&out.stdout);
    assert!(
        nm_text.contains("Provider_GetHost"),
        "providers_shared does not export Provider_GetHost"
    );

    let out = Command::new("env")
        .arg(format!("LD_PRELOAD={}", shared.display()))
        .arg("ldd")
        .arg("-r")
        .arg(&trt)
        .output()
        .expect("run ldd -r");
    assert!(out.status.success(), "ldd -r failed for {}", trt.display());
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !text.contains("undefined symbol: Provider_GetHost"),
        "Provider_GetHost unresolved for {}",
        trt.display()
    );
}

#[test]
#[ignore = "requires model + full TensorRT runtime"]
fn ort_tensorrt_ep_registration_smoke() {
    let model = env::var("RAVE_TEST_ONNX_MODEL").expect("set RAVE_TEST_ONNX_MODEL");
    let shared = provider_candidates("libonnxruntime_providers_shared.so")
        .into_iter()
        .next()
        .expect("providers_shared not found");

    dlopen_path(&shared, RTLD_NOW | RTLD_GLOBAL).expect("preload providers_shared");

    let ep = TensorRTExecutionProvider::default()
        .build()
        .error_on_failure();
    let _session = Session::builder()
        .expect("session builder")
        .with_execution_providers([ep])
        .expect("register trt ep")
        .commit_from_file(model)
        .expect("create ORT session with TRT EP");
}
