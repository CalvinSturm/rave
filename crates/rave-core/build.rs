use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-changed=build.rs");

    if cfg!(target_os = "windows") {
        let cuda_path = env::var("CUDA_PATH")
            .expect("CUDA_PATH must be set (e.g., C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x)");
        let cuda_lib_dir = PathBuf::from(cuda_path).join("lib").join("x64");
        if !cuda_lib_dir.exists() {
            panic!(
                "CRITICAL: CUDA library directory not found at {}",
                cuda_lib_dir.display()
            );
        }
        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=cuda");
    }
}
