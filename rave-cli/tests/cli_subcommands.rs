use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("rave_cli_{label}_{}_{}", std::process::id(), nanos));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn write_dummy_file(path: &PathBuf) {
    fs::write(path, b"dummy").expect("write dummy file");
}

#[test]
fn help_lists_subcommands() {
    let output = Command::new(env!("CARGO_BIN_EXE_rave"))
        .arg("help")
        .output()
        .expect("run rave help");

    assert!(
        output.status.success(),
        "rave help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("upscale"), "missing upscale in help output");
    assert!(
        stdout.contains("benchmark"),
        "missing benchmark in help output"
    );
    assert!(stdout.contains("probe"), "missing probe in help output");
    assert!(stdout.contains("devices"), "missing devices in help output");
}

#[test]
fn benchmark_help_lists_skip_encode_and_json_out() {
    let output = Command::new(env!("CARGO_BIN_EXE_rave"))
        .args(["benchmark", "--help"])
        .output()
        .expect("run rave benchmark --help");

    assert!(
        output.status.success(),
        "benchmark --help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--skip-encode"),
        "missing --skip-encode in benchmark help"
    );
    assert!(
        stdout.contains("--json-out"),
        "missing --json-out in benchmark help"
    );
    assert!(
        stdout.contains("--json"),
        "missing --json in benchmark help"
    );
    assert!(
        stdout.contains("--progress"),
        "missing --progress in benchmark help"
    );
    assert!(
        stdout.contains("--jsonl"),
        "missing --jsonl in benchmark help"
    );
}

#[test]
fn probe_help_lists_all_and_json() {
    let output = Command::new(env!("CARGO_BIN_EXE_rave"))
        .args(["probe", "--help"])
        .output()
        .expect("run rave probe --help");

    assert!(
        output.status.success(),
        "probe --help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--all"), "missing --all in probe help");
    assert!(stdout.contains("--json"), "missing --json in probe help");
}

#[test]
fn devices_help_lists_json() {
    let output = Command::new(env!("CARGO_BIN_EXE_rave"))
        .args(["devices", "--help"])
        .output()
        .expect("run rave devices --help");

    assert!(
        output.status.success(),
        "devices --help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--json"), "missing --json in devices help");
}

#[test]
fn upscale_dry_run_accepts_subcommand_args() {
    let dir = unique_temp_dir("upscale");
    let input = dir.join("input.265");
    let model = dir.join("model.onnx");
    let output_path = dir.join("output.265");
    write_dummy_file(&input);
    write_dummy_file(&model);

    let output = Command::new(env!("CARGO_BIN_EXE_rave"))
        .args([
            "upscale",
            "--input",
            input.to_str().expect("utf8 input"),
            "--output",
            output_path.to_str().expect("utf8 output"),
            "--model",
            model.to_str().expect("utf8 model"),
            "--dry-run",
        ])
        .output()
        .expect("run rave upscale --dry-run");

    assert!(
        output.status.success(),
        "upscale dry-run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("dry-run: command=upscale"),
        "unexpected dry-run output: {stdout}"
    );
}

#[test]
fn upscale_help_lists_progress_and_jsonl() {
    let output = Command::new(env!("CARGO_BIN_EXE_rave"))
        .args(["upscale", "--help"])
        .output()
        .expect("run rave upscale --help");

    assert!(
        output.status.success(),
        "upscale --help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--progress"),
        "missing --progress in upscale help"
    );
    assert!(
        stdout.contains("--jsonl"),
        "missing --jsonl in upscale help"
    );
}

#[test]
fn benchmark_dry_run_emits_valid_json_shape() {
    let dir = unique_temp_dir("benchmark");
    let input = dir.join("input.265");
    let model = dir.join("model.onnx");
    let json_out = dir.join("bench.json");
    write_dummy_file(&input);
    write_dummy_file(&model);

    let output = Command::new(env!("CARGO_BIN_EXE_rave"))
        .args([
            "benchmark",
            "--input",
            input.to_str().expect("utf8 input"),
            "--model",
            model.to_str().expect("utf8 model"),
            "--json-out",
            json_out.to_str().expect("utf8 json out"),
            "--json",
            "--dry-run",
        ])
        .output()
        .expect("run rave benchmark --dry-run");

    assert!(
        output.status.success(),
        "benchmark dry-run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("benchmark output is not valid JSON: {e}"));
    assert!(
        value.get("fps").and_then(|v| v.as_f64()).is_some(),
        "missing numeric fps field"
    );
    let stages = value.get("stages").expect("missing stages object");
    assert!(stages.get("decode").is_some(), "missing stages.decode");
    assert!(stages.get("infer").is_some(), "missing stages.infer");
    assert!(stages.get("encode").is_some(), "missing stages.encode");

    let json_out_value: serde_json::Value = serde_json::from_slice(
        &fs::read(&json_out).expect("benchmark --json-out file should exist"),
    )
    .expect("benchmark --json-out should contain valid JSON");
    assert!(
        json_out_value.get("stages").is_some(),
        "missing stages in benchmark --json-out payload"
    );
}

#[test]
fn benchmark_dry_run_without_json_is_human_readable() {
    let dir = unique_temp_dir("benchmark_human");
    let input = dir.join("input.265");
    let model = dir.join("model.onnx");
    write_dummy_file(&input);
    write_dummy_file(&model);

    let output = Command::new(env!("CARGO_BIN_EXE_rave"))
        .args([
            "benchmark",
            "--input",
            input.to_str().expect("utf8 input"),
            "--model",
            model.to_str().expect("utf8 model"),
            "--dry-run",
        ])
        .output()
        .expect("run rave benchmark --dry-run");

    assert!(
        output.status.success(),
        "benchmark dry-run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with("benchmark: "),
        "expected human-readable benchmark summary, got: {stdout}"
    );
}

#[test]
fn benchmark_dry_run_accepts_jsonl_progress_flag() {
    let dir = unique_temp_dir("benchmark_jsonl_progress");
    let input = dir.join("input.265");
    let model = dir.join("model.onnx");
    write_dummy_file(&input);
    write_dummy_file(&model);

    let output = Command::new(env!("CARGO_BIN_EXE_rave"))
        .args([
            "benchmark",
            "--input",
            input.to_str().expect("utf8 input"),
            "--model",
            model.to_str().expect("utf8 model"),
            "--progress",
            "jsonl",
            "--dry-run",
        ])
        .output()
        .expect("run rave benchmark --progress jsonl --dry-run");

    assert!(
        output.status.success(),
        "benchmark dry-run with progress flag failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with("benchmark: "),
        "expected human-readable benchmark summary, got: {stdout}"
    );
}

#[test]
fn upscale_dry_run_json_emits_valid_json_shape() {
    let dir = unique_temp_dir("upscale_json");
    let input = dir.join("input.265");
    let model = dir.join("model.onnx");
    let output_path = dir.join("output.265");
    write_dummy_file(&input);
    write_dummy_file(&model);

    let output = Command::new(env!("CARGO_BIN_EXE_rave"))
        .args([
            "upscale",
            "--input",
            input.to_str().expect("utf8 input"),
            "--output",
            output_path.to_str().expect("utf8 output"),
            "--model",
            model.to_str().expect("utf8 model"),
            "--json",
            "--dry-run",
        ])
        .output()
        .expect("run rave upscale --json --dry-run");

    assert!(
        output.status.success(),
        "upscale json dry-run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("upscale json dry-run output is not valid JSON: {e}"));
    assert_eq!(
        value.get("command").and_then(|v| v.as_str()),
        Some("upscale"),
        "missing command=upscale field"
    );
    assert_eq!(
        value.get("dry_run").and_then(|v| v.as_bool()),
        Some(true),
        "missing dry_run=true field"
    );
    assert!(
        value.get("width").and_then(|v| v.as_u64()).is_some(),
        "missing width field"
    );
    assert!(
        value.get("height").and_then(|v| v.as_u64()).is_some(),
        "missing height field"
    );
}
