//! Build script — locate CUDA toolkit and link nvcuvid + nvEncodeAPI.
//!
//! Requires `CUDA_PATH` env var pointing to the CUDA toolkit root.

use std::env;
use std::path::PathBuf;

fn main() {
    // Only re-run if these change.
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-changed=build.rs");

    let cuda_path = env::var("CUDA_PATH")
        .expect("CUDA_PATH env var must be set (e.g., C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x)");

    let root = PathBuf::from(cuda_path);

    // 1. Library Search Paths
    let lib_dir = if cfg!(target_os = "windows") {
        root.join("lib").join("x64")
    } else {
        root.join("lib64")
    };

    if !lib_dir.exists() {
        panic!(
            "CRITICAL: CUDA library directory not found at {}",
            lib_dir.display()
        );
    }
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // 2. Header Search Paths (Propagate to dependencies/bindgen)
    let include_dir = root.join("include");
    if !include_dir.exists() {
        panic!(
            "CRITICAL: CUDA include directory not found at {}",
            include_dir.display()
        );
    }
    // This allows other crates or bindgen to find nvcuvid.h / nvEncodeAPI.h
    println!("cargo:include={}", include_dir.display());

    // 3. Link nvcuvid (NVDEC)
    // Verify .lib existence on Windows to provide better error messages
    if cfg!(target_os = "windows") {
        let nvcuvid_lib = lib_dir.join("nvcuvid.lib");
        if !nvcuvid_lib.exists() {
            panic!("MISSING SDK: nvcuvid.lib not found in {}. Did you copy it from the Video Codec SDK?", lib_dir.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=nvcuvid");

    // 4. Link nvEncodeAPI (NVENC)
    if cfg!(target_os = "windows") {
        let nvenc_lib = lib_dir.join("nvencodeapi.lib");
        if !nvenc_lib.exists() {
            panic!("MISSING SDK: nvencodeapi.lib not found in {}. Did you copy it from the Video Codec SDK?", lib_dir.display());
        }
        println!("cargo:rustc-link-lib=dylib=nvencodeapi");
    } else {
        println!("cargo:rustc-link-lib=dylib=nvidia-encode");
    }

    // 5. Link CUDA Driver API
    // Used for cuEventRecord, cuStreamWaitEvent, and raw pointer manipulation
    println!("cargo:rustc-link-lib=dylib=cuda");
}
