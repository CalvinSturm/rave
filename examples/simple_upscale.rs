// 10-line minimal end-to-end upscale example.
fn main() {
    let status = std::process::Command::new("cargo")
        .args([
            "run",
            "-p",
            "rave-cli",
            "--bin",
            "rave",
            "--locked",
            "--",
            "upscale",
            "--input",
            "legacy/engine-v2/test_videos/Input.mp4",
            "--output",
            "/tmp/rave_simple_upscale.mp4",
            "--model",
            "legacy/engine-v2/models/4xNomos2_hq_dat2_fp32.onnx",
        ])
        .status()
        .expect("failed to run rave upscale");
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
}
