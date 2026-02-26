#![allow(missing_docs)]
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

#[cfg(target_os = "linux")]
fn find_linux_cuda_root() -> Option<PathBuf> {
    let mut candidates = vec![PathBuf::from("/usr/local/cuda")];
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        let mut versioned = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("cuda-"))
            })
            .collect::<Vec<_>>();
        versioned.sort();
        versioned.reverse();
        candidates.extend(versioned);
    }

    candidates.into_iter().find(|root| root.exists())
}

#[cfg(not(target_os = "linux"))]
fn find_linux_cuda_root() -> Option<PathBuf> {
    None
}

fn resolve_cuda_root() -> Option<PathBuf> {
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        return Some(PathBuf::from(cuda_path));
    }

    if let Some(root) = find_linux_cuda_root() {
        println!(
            "cargo:warning=CUDA_PATH is unset; using discovered CUDA root at {}",
            root.display()
        );
        return Some(root);
    }

    None
}

fn resolve_nvcodec_dir(manifest_dir: &PathBuf) -> Option<PathBuf> {
    let rave_root = manifest_dir.parent()?.parent()?;
    let mut candidates = vec![
        // Vendored inside the rave repo.
        rave_root.join("third_party").join("nvcodec"),
    ];
    // Parent app layout: <app>/third_party/{rave,nvcodec,ffmpeg}.
    if let Some(parent_third_party) = rave_root.parent() {
        candidates.push(parent_third_party.join("nvcodec"));
    }

    candidates.into_iter().find(|dir| {
        dir.exists() && dir.join("nvcuvid.lib").exists() && dir.join("nvencodeapi.lib").exists()
    })
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(rave_nvcodec_stub)");
    println!("cargo:rustc-check-cfg=cfg(docsrs)");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");
    println!("cargo:rerun-if-changed=build.rs");

    if env::var_os("DOCS_RS").is_some() {
        println!("cargo:warning=DOCS_RS detected; building rave-nvcodec in stub mode");
        println!("cargo:rustc-cfg=rave_nvcodec_stub");
        return;
    }

    // ── CUDA Toolkit ────────────────────────────────────────────────────────

    let Some(cuda_root) = resolve_cuda_root() else {
        if cfg!(target_os = "linux") {
            println!(
                "cargo:warning=CUDA toolkit not found (CUDA_PATH unset and /usr/local/cuda* missing); building rave-nvcodec in stub mode"
            );
            println!("cargo:rustc-cfg=rave_nvcodec_stub");
            return;
        }
        panic!(
            "CUDA_PATH env var must be set (e.g., C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x)"
        );
    };

    let cuda_lib_dir = if cfg!(target_os = "windows") {
        cuda_root.join("lib").join("x64")
    } else {
        let wsl_style = cuda_root.join("targets").join("x86_64-linux").join("lib");
        if wsl_style.exists() {
            wsl_style
        } else {
            cuda_root.join("lib64")
        }
    };

    if !cuda_lib_dir.exists() {
        panic!(
            "CRITICAL: CUDA library directory not found at {}",
            cuda_lib_dir.display()
        );
    }
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());

    // On Linux (especially WSL2), driver libs are frequently outside CUDA_PATH.
    // Add common locations so rust-lld can resolve -lcuda/-lnvcuvid/-lnvidia-encode.
    if cfg!(target_os = "linux") {
        for extra in ["/usr/lib/wsl/lib", "/usr/local/lib/wsl-nvidia"] {
            let p = PathBuf::from(extra);
            if p.exists() {
                println!("cargo:rustc-link-search=native={}", p.display());
            }
        }
    }

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
    let nvcodec_dir = resolve_nvcodec_dir(&manifest_dir);

    if cfg!(target_os = "windows") {
        if let Some(nvcodec_dir) = nvcodec_dir {
            println!("cargo:rustc-link-search=native={}", nvcodec_dir.display());
        } else {
            // Fallback: expect libs in CUDA toolkit directory
            println!(
                "cargo:warning=Video Codec SDK libs not found in expected nvcodec locations. Falling back to CUDA lib dir."
            );
        }
    } else if let Some(nvcodec_dir) = nvcodec_dir {
        // Linux toolchains consume .so/.a, but adding this path is harmless and
        // supports repos that vendor Linux NVCodec artifacts in third_party/nvcodec.
        println!("cargo:rustc-link-search=native={}", nvcodec_dir.display());
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
