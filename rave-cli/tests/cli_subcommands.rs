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

fn assert_schema_version(value: &serde_json::Value) {
    assert_eq!(
        value.get("schema_version").and_then(|v| v.as_u64()),
        Some(1),
        "missing schema_version=1 field"
    );
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
fn probe_json_emits_schema_and_command_fields() {
    let output = Command::new(env!("CARGO_BIN_EXE_rave"))
        .args(["probe", "--json"])
        .output()
        .expect("run rave probe --json");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("\u{1b}["),
        "stderr should not include ANSI escapes when not a TTY: {stderr}"
    );

    let value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("probe --json stdout is not JSON: {e}"));
    assert_schema_version(&value);
    assert_eq!(
        value.get("command").and_then(|v| v.as_str()),
        Some("probe"),
        "missing command=probe field"
    );
    assert!(
        value.get("ok").and_then(|v| v.as_bool()).is_some(),
        "missing boolean ok field"
    );
}

#[test]
fn devices_json_emits_schema_and_command_fields() {
    let output = Command::new(env!("CARGO_BIN_EXE_rave"))
        .args(["devices", "--json"])
        .output()
        .expect("run rave devices --json");

    let value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("devices --json stdout is not JSON: {e}"));
    assert_schema_version(&value);
    assert_eq!(
        value.get("command").and_then(|v| v.as_str()),
        Some("devices"),
        "missing command=devices field"
    );
    assert!(
        value.get("ok").and_then(|v| v.as_bool()).is_some(),
        "missing boolean ok field"
    );
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
    assert_schema_version(&value);
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
    assert_schema_version(&json_out_value);
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
fn benchmark_dry_run_json_mode_is_clean_stdout_even_with_progress_flags() {
    let dir = unique_temp_dir("benchmark_json_stdout_clean");
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
            "--json",
            "--json-out",
            json_out.to_str().expect("utf8 json out"),
            "--progress",
            "jsonl",
            "--jsonl",
            "--dry-run",
        ])
        .output()
        .expect("run rave benchmark with json + progress flags");

    assert!(
        output.status.success(),
        "benchmark dry-run with json/progress flags failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("benchmark stdout is not clean JSON: {e}"));
    assert_schema_version(&stdout_value);
    assert_eq!(
        stdout_value.get("command").and_then(|v| v.as_str()),
        Some("benchmark")
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("\"type\":\"progress\"") && !stderr.contains("progress: command="),
        "dry-run should not emit progress records, got stderr: {stderr}"
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
    assert_schema_version(&value);
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

#[test]
fn upscale_dry_run_json_mode_is_clean_stdout_even_with_progress_flags() {
    let dir = unique_temp_dir("upscale_json_stdout_clean");
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
            "--progress",
            "jsonl",
            "--jsonl",
            "--dry-run",
        ])
        .output()
        .expect("run rave upscale with json + progress flags");

    assert!(
        output.status.success(),
        "upscale dry-run with json/progress flags failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("upscale stdout is not clean JSON: {e}"));
    assert_schema_version(&stdout_value);
    assert_eq!(
        stdout_value.get("command").and_then(|v| v.as_str()),
        Some("upscale")
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("\"type\":\"progress\"") && !stderr.contains("progress: command="),
        "dry-run should not emit progress records, got stderr: {stderr}"
    );
}

#[test]
fn benchmark_json_mode_emits_structured_error_on_failure() {
    let dir = unique_temp_dir("benchmark_json_error");
    let missing_input = dir.join("missing.265");
    let missing_model = dir.join("missing.onnx");

    let output = Command::new(env!("CARGO_BIN_EXE_rave"))
        .args([
            "benchmark",
            "--input",
            missing_input.to_str().expect("utf8 input"),
            "--model",
            missing_model.to_str().expect("utf8 model"),
            "--json",
        ])
        .output()
        .expect("run rave benchmark --json with missing paths");

    assert!(
        !output.status.success(),
        "benchmark should fail for missing paths"
    );
    let stdout_value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("benchmark error stdout is not JSON: {e}"));
    assert_schema_version(&stdout_value);
    assert_eq!(
        stdout_value.get("command").and_then(|v| v.as_str()),
        Some("benchmark")
    );
    assert_eq!(
        stdout_value.get("ok").and_then(|v| v.as_bool()),
        Some(false)
    );
    assert!(
        stdout_value.get("error").and_then(|v| v.as_str()).is_some(),
        "missing error field in benchmark json error payload"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("Command failed"),
        "json mode should not emit default error line on stderr: {stderr}"
    );
}

#[test]
fn upscale_json_mode_emits_structured_error_on_failure() {
    let dir = unique_temp_dir("upscale_json_error");
    let missing_input = dir.join("missing.265");
    let missing_model = dir.join("missing.onnx");
    let output_path = dir.join("out.265");

    let output = Command::new(env!("CARGO_BIN_EXE_rave"))
        .args([
            "upscale",
            "--input",
            missing_input.to_str().expect("utf8 input"),
            "--output",
            output_path.to_str().expect("utf8 output"),
            "--model",
            missing_model.to_str().expect("utf8 model"),
            "--json",
        ])
        .output()
        .expect("run rave upscale --json with missing paths");

    assert!(
        !output.status.success(),
        "upscale should fail for missing paths"
    );
    let stdout_value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("upscale error stdout is not JSON: {e}"));
    assert_schema_version(&stdout_value);
    assert_eq!(
        stdout_value.get("command").and_then(|v| v.as_str()),
        Some("upscale")
    );
    assert_eq!(
        stdout_value.get("ok").and_then(|v| v.as_bool()),
        Some(false)
    );
    assert!(
        stdout_value.get("error").and_then(|v| v.as_str()).is_some(),
        "missing error field in upscale json error payload"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("Command failed"),
        "json mode should not emit default error line on stderr: {stderr}"
    );
}
