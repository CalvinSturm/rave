//! Build script — locate CUDA toolkit, Video Codec SDK, and FFmpeg.
//!
//! Required environment variables:
//!   CUDA_PATH  — CUDA toolkit root (set by NVIDIA installer)
//!   FFMPEG_DIR — FFmpeg root with lib/, include/, bin/ (set in .cargo/config.toml)
//!
//! Video Codec SDK libs (nvcuvid.lib, nvencodeapi.lib) are resolved from:
//!   1. ../third_party/nvcodec (bundled in repo)
//!   2. CUDA_PATH/lib/x64 (fallback)

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");
    println!("cargo:rerun-if-changed=build.rs");

    // ── CUDA Toolkit ────────────────────────────────────────────────────────

    let cuda_path = env::var("CUDA_PATH")
        .expect("CUDA_PATH env var must be set (e.g., C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x)");

    let cuda_root = PathBuf::from(cuda_path);

    let cuda_lib_dir = if cfg!(target_os = "windows") {
        cuda_root.join("lib").join("x64")
    } else {
        cuda_root.join("lib64")
    };

    if !cuda_lib_dir.exists() {
        panic!(
            "CRITICAL: CUDA library directory not found at {}",
            cuda_lib_dir.display()
        );
    }
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());

    let cuda_include_dir = cuda_root.join("include");
    if !cuda_include_dir.exists() {
        panic!(
            "CRITICAL: CUDA include directory not found at {}",
            cuda_include_dir.display()
        );
    }
    println!("cargo:include={}", cuda_include_dir.display());

    // Link CUDA Driver API
    println!("cargo:rustc-link-lib=dylib=cuda");

    // ── Video Codec SDK (nvcuvid + nvEncodeAPI) ─────────────────────────────

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let nvcodec_dir = manifest_dir
        .parent()
        .unwrap()
        .join("third_party")
        .join("nvcodec");

    if nvcodec_dir.exists()
        && nvcodec_dir.join("nvcuvid.lib").exists()
        && nvcodec_dir.join("nvencodeapi.lib").exists()
    {
        println!("cargo:rustc-link-search=native={}", nvcodec_dir.display());
    } else {
        // Fallback: expect libs in CUDA toolkit directory
        println!(
            "cargo:warning=Video Codec SDK libs not found in {}. Falling back to CUDA lib dir.",
            nvcodec_dir.display()
        );
    }

    println!("cargo:rustc-link-lib=dylib=nvcuvid");
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=dylib=nvencodeapi");
    } else {
        println!("cargo:rustc-link-lib=dylib=nvidia-encode");
    }

    // ── FFmpeg ───────────────────────────────────────────────────────────────
    // FFMPEG_DIR is set in .cargo/config.toml → ffmpeg-sys-next picks it up.
    // We also add the lib path so the linker can resolve avcodec.lib etc.

    if let Ok(ffmpeg_dir) = env::var("FFMPEG_DIR") {
        let ffmpeg_lib = PathBuf::from(&ffmpeg_dir).join("lib");
        if ffmpeg_lib.exists() {
            println!("cargo:rustc-link-search=native={}", ffmpeg_lib.display());
        }
    }
}
