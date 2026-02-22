use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "rave_cli_stream_{label}_{}_{}",
        std::process::id(),
        nanos
    ));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn write_dummy_file(path: &Path) {
    fs::write(path, b"dummy").expect("write dummy file");
}

fn nonempty_lines(s: &str) -> Vec<&str> {
    s.lines().filter(|line| !line.trim().is_empty()).collect()
}

fn assert_single_stdout_json(stdout: &[u8], command: &str, ok: bool) {
    let stdout_s = String::from_utf8_lossy(stdout);
    let lines = nonempty_lines(&stdout_s);
    assert_eq!(
        lines.len(),
        1,
        "stdout must contain exactly one non-empty line, got {}:\n{}",
        lines.len(),
        stdout_s
    );
    assert!(
        !stdout_s.contains("\"type\":\"progress\""),
        "stdout must not contain progress records: {stdout_s}"
    );
    let value: serde_json::Value = serde_json::from_str(lines[0])
        .unwrap_or_else(|e| panic!("stdout is not JSON object: {e}\n{stdout_s}"));
    assert_eq!(
        value.get("command").and_then(|v| v.as_str()),
        Some(command),
        "unexpected command field in stdout JSON: {value}"
    );
    assert_eq!(
        value.get("ok").and_then(|v| v.as_bool()),
        Some(ok),
        "unexpected ok field in stdout JSON: {value}"
    );
}

fn assert_stderr_has_progress_jsonl(stderr: &[u8], command: &str) {
    let stderr_s = String::from_utf8_lossy(stderr);
    let lines = nonempty_lines(&stderr_s);
    assert!(
        !lines.is_empty(),
        "stderr must contain progress output, got empty stderr"
    );

    let mut progress_lines = 0usize;
    for line in &lines {
        let value: serde_json::Value = serde_json::from_str(line).unwrap_or_else(|e| {
            panic!("stderr line is not JSONL progress: {e}\nline={line}\nstderr={stderr_s}")
        });
        assert_eq!(
            value.get("type").and_then(|v| v.as_str()),
            Some("progress"),
            "stderr JSONL line is not a progress record: {value}"
        );
        assert_eq!(
            value.get("command").and_then(|v| v.as_str()),
            Some(command),
            "stderr progress command mismatch: {value}"
        );
        progress_lines += 1;
    }
    assert!(progress_lines >= 1, "expected at least one progress line");
    assert!(
        !stderr_s.contains("\"ok\":true") || stderr_s.contains("\"type\":\"progress\""),
        "stderr should not contain final result JSON: {stderr_s}"
    );
}

fn mock_command() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rave"));
    cmd.env("RAVE_MOCK_RUN", "1")
        .env("RAVE_PROGRESS_TICK_MS", "10");
    cmd
}

#[test]
fn upscale_json_stdout_is_clean_and_stderr_has_progress() {
    let dir = unique_temp_dir("upscale");
    let input = dir.join("input.265");
    let model = dir.join("model.onnx");
    let output_path = dir.join("output.265");
    write_dummy_file(&input);
    write_dummy_file(&model);

    let output = mock_command()
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
        ])
        .output()
        .expect("run mock rave upscale");

    assert!(
        output.status.success(),
        "mock upscale failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_single_stdout_json(&output.stdout, "upscale", true);
    assert_stderr_has_progress_jsonl(&output.stderr, "upscale");
}

#[test]
fn benchmark_json_stdout_is_clean_and_stderr_has_progress() {
    let dir = unique_temp_dir("benchmark");
    let input = dir.join("input.265");
    let model = dir.join("model.onnx");
    write_dummy_file(&input);
    write_dummy_file(&model);

    let output = mock_command()
        .args([
            "benchmark",
            "--input",
            input.to_str().expect("utf8 input"),
            "--model",
            model.to_str().expect("utf8 model"),
            "--json",
            "--progress",
            "jsonl",
            "--skip-encode",
        ])
        .output()
        .expect("run mock rave benchmark");

    assert!(
        output.status.success(),
        "mock benchmark failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_single_stdout_json(&output.stdout, "benchmark", true);
    assert_stderr_has_progress_jsonl(&output.stderr, "benchmark");
}

#[test]
fn validate_json_stdout_is_clean_and_stderr_has_progress() {
    let dir = unique_temp_dir("validate");
    let input = dir.join("input.265");
    let model = dir.join("model.onnx");
    let output_path = dir.join("output.265");
    write_dummy_file(&input);
    write_dummy_file(&model);

    let output = mock_command()
        .args([
            "validate",
            "--profile",
            "dev",
            "--input",
            input.to_str().expect("utf8 input"),
            "--model",
            model.to_str().expect("utf8 model"),
            "--output",
            output_path.to_str().expect("utf8 output"),
            "--json",
            "--progress",
            "jsonl",
        ])
        .output()
        .expect("run mock rave validate");

    assert!(
        output.status.success(),
        "mock validate failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_single_stdout_json(&output.stdout, "validate", true);
    assert_stderr_has_progress_jsonl(&output.stderr, "validate");
}

#[test]
fn upscale_json_error_is_single_object_on_stdout() {
    let dir = unique_temp_dir("upscale_err");
    let input = dir.join("input.265");
    let model = dir.join("model.onnx");
    let output_path = dir.join("output.265");
    write_dummy_file(&input);
    write_dummy_file(&model);

    let output = Command::new(env!("CARGO_BIN_EXE_rave"))
        .env("RAVE_MOCK_RUN", "1")
        .env("RAVE_MOCK_FAIL", "1")
        .env("RAVE_PROGRESS_TICK_MS", "10")
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
        ])
        .output()
        .expect("run mock rave upscale failure");

    assert!(
        !output.status.success(),
        "mock upscale failure unexpectedly succeeded"
    );
    assert_single_stdout_json(&output.stdout, "upscale", false);
    assert_stderr_has_progress_jsonl(&output.stderr, "upscale");
}

#[test]
fn upscale_json_rejects_micro_batching_in_mock_mode() {
    let dir = unique_temp_dir("upscale_batch_guard");
    let input = dir.join("input.265");
    let model = dir.join("model.onnx");
    let output_path = dir.join("output.265");
    let graph_path = dir.join("graph.json");
    write_dummy_file(&input);
    write_dummy_file(&model);
    let graph = serde_json::json!({
        "graph_schema_version": 1,
        "stages": [{
            "kind": "enhance",
            "id": 1,
            "config": {
                "model_path": model,
                "precision_policy": "fp16",
                "batch_config": {
                    "max_batch": 2,
                    "latency_deadline_us": 8000
                },
                "scale": 2
            }
        }]
    });
    fs::write(
        &graph_path,
        serde_json::to_vec(&graph).expect("serialize graph"),
    )
    .expect("write graph");

    let output = mock_command()
        .args([
            "upscale",
            "--input",
            input.to_str().expect("utf8 input"),
            "--output",
            output_path.to_str().expect("utf8 output"),
            "--model",
            model.to_str().expect("utf8 model"),
            "--graph",
            graph_path.to_str().expect("utf8 graph"),
            "--json",
            "--progress",
            "jsonl",
        ])
        .output()
        .expect("run mock rave upscale with unsupported batch config");

    assert!(
        !output.status.success(),
        "mock upscale with max_batch=2 unexpectedly succeeded"
    );
    assert_single_stdout_json(&output.stdout, "upscale", false);
    let stdout_s = String::from_utf8_lossy(&output.stdout);
    let lines = nonempty_lines(&stdout_s);
    let value: serde_json::Value =
        serde_json::from_str(lines[0]).expect("stdout JSON should parse");
    let error = value
        .get("error")
        .and_then(|v| v.as_str())
        .expect("error field must be present");
    assert!(
        error.contains("max_batch"),
        "unexpected error message: {error}"
    );
    assert!(
        error.contains("not implemented"),
        "unexpected error message: {error}"
    );
}

#[test]
fn validate_mock_production_strict_enforces_host_copy_audit_availability() {
    let output = mock_command()
        .args([
            "validate",
            "--profile",
            "production_strict",
            "--json",
            "--progress",
            "jsonl",
        ])
        .output()
        .expect("run mock rave validate production_strict");

    let strict_ok = cfg!(feature = "audit-no-host-copies");
    assert_eq!(
        output.status.success(),
        strict_ok,
        "unexpected strict validate status (stdout={}, stderr={})",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_single_stdout_json(&output.stdout, "validate", strict_ok);
    assert_stderr_has_progress_jsonl(&output.stderr, "validate");

    let stdout_s = String::from_utf8_lossy(&output.stdout);
    let lines = nonempty_lines(&stdout_s);
    let value: serde_json::Value =
        serde_json::from_str(lines[0]).expect("stdout JSON should parse");
    if strict_ok {
        assert_eq!(
            value
                .get("host_copy_audit_enabled")
                .and_then(|v| v.as_bool()),
            Some(true)
        );
        assert!(
            value
                .get("host_copy_audit_disable_reason")
                .is_some_and(|v| v.is_null()),
            "disable reason should be null when audit is enabled: {value}"
        );
    } else {
        let error = value
            .get("error")
            .and_then(|v| v.as_str())
            .expect("error field must be present");
        assert!(error.contains("strict no-host-copies"));
        assert!(error.contains("feature_disabled"));
    }
}
