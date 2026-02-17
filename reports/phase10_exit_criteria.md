# Phase 10 Exit Criteria

Date: 2026-02-17T23:06:58Z

## Command
```bash
cargo fmt --check
```

### Output
```text

[exit_code]=0
```

## Command
```bash
cargo clippy --workspace --all-targets -- -D warnings
```

### Output
```text
    Checking rave-tensorrt v2.0.0 (/home/calvin/src/rave/crates/rave-tensorrt)
    Checking rave-cuda v2.0.0 (/home/calvin/src/rave/crates/rave-cuda)
    Checking rave-ffmpeg v2.0.0 (/home/calvin/src/rave/crates/rave-ffmpeg)
    Checking rave-nvcodec v2.0.0 (/home/calvin/src/rave/crates/rave-nvcodec)
error: casting to the same type is unnecessary (`u64` -> `u64`)
   --> crates/rave-cuda/src/kernels.rs:376:23
    |
376 |         let out_ptr = *output_buf.device_ptr() as u64;
    |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `*output_buf.device_ptr()`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast
    = note: `-D clippy::unnecessary-cast` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::unnecessary_cast)]`

error: casting to the same type is unnecessary (`u64` -> `u64`)
   --> crates/rave-cuda/src/kernels.rs:442:23
    |
442 |         let out_ptr = *output_buf.device_ptr() as u64;
    |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `*output_buf.device_ptr()`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast

error: casting to the same type is unnecessary (`u64` -> `u64`)
   --> crates/rave-cuda/src/kernels.rs:497:23
    |
497 |         let out_ptr = *output_buf.device_ptr() as u64;
    |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `*output_buf.device_ptr()`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast

error: casting to the same type is unnecessary (`u64` -> `u64`)
   --> crates/rave-cuda/src/kernels.rs:543:23
    |
543 |         let out_ptr = *output_buf.device_ptr() as u64;
    |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `*output_buf.device_ptr()`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast

error: casting to the same type is unnecessary (`u64` -> `u64`)
   --> crates/rave-cuda/src/kernels.rs:592:21
    |
592 |         let y_ptr = *output_buf.device_ptr() as u64;
    |                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `*output_buf.device_ptr()`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast

error: manually reimplementing `div_ceil`
   --> crates/rave-cuda/src/kernels.rs:947:9
    |
947 |         (width + block.0 - 1) / block.0,
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: consider using `.div_ceil()`: `width.div_ceil(block.0)`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#manual_div_ceil
    = note: `-D clippy::manual-div-ceil` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::manual_div_ceil)]`

error: manually reimplementing `div_ceil`
   --> crates/rave-cuda/src/kernels.rs:948:9
    |
948 |         (height + block.1 - 1) / block.1,
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: consider using `.div_ceil()`: `height.div_ceil(block.1)`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#manual_div_ceil

error: manually reimplementing `div_ceil`
   --> crates/rave-cuda/src/kernels.rs:964:17
    |
964 |     let grid = ((count as u32 + block - 1) / block, 1, 1);
    |                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: consider using `.div_ceil()`: `(count as u32).div_ceil(block)`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#manual_div_ceil

error: this public function might dereference a raw pointer but is not marked `unsafe`
  --> crates/rave-cuda/src/stream.rs:19:48
   |
19 |             sys::cuStreamWaitEvent(raw_stream, event, 0),
   |                                                ^^^^^
   |
   = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#not_unsafe_ptr_arg_deref
   = note: `#[deny(clippy::not_unsafe_ptr_arg_deref)]` on by default

error: could not compile `rave-cuda` (lib) due to 9 previous errors
warning: build failed, waiting for other jobs to finish...
error: could not compile `rave-cuda` (lib test) due to 9 previous errors
error: items after a test module
   --> crates/rave-ffmpeg/src/ffmpeg_demuxer.rs:312:1
    |
312 | mod tests {
    | ^^^^^^^^^
...
323 | impl Drop for FfmpegDemuxer {
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#items_after_test_module
    = note: `-D clippy::items-after-test-module` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::items_after_test_module)]`
    = help: move the items to before the test module was defined

error: manual implementation of `.is_multiple_of()`
   --> crates/rave-ffmpeg/src/ffmpeg_muxer.rs:182:12
    |
182 |         if self.packet_counter % 100 == 0 {
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: replace with: `self.packet_counter.is_multiple_of(100)`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#manual_is_multiple_of
    = note: `-D clippy::manual-is-multiple-of` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::manual_is_multiple_of)]`

error: manual implementation of `.is_multiple_of()`
  --> crates/rave-ffmpeg/src/file_sink.rs:66:12
   |
66 |         if self.packets_written % 100 == 0 {
   |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: replace with: `self.packets_written.is_multiple_of(100)`
   |
   = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#manual_is_multiple_of

error: could not compile `rave-ffmpeg` (lib test) due to 3 previous errors
error: could not compile `rave-ffmpeg` (lib) due to 2 previous errors
error: this public function might dereference a raw pointer but is not marked `unsafe`
   --> crates/rave-nvcodec/src/nvdec.rs:634:48
    |
634 |         check_cu(cuStreamWaitEvent(raw_stream, event, 0), "cuStreamWaitEvent")?;
    |                                                ^^^^^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#not_unsafe_ptr_arg_deref
    = note: `#[deny(clippy::not_unsafe_ptr_arg_deref)]` on by default

error: this `impl` can be derived
   --> crates/rave-tensorrt/src/tensorrt.rs:141:1
    |
141 | / impl Default for PrecisionPolicy {
142 | |     fn default() -> Self {
143 | |         PrecisionPolicy::Fp16
144 | |     }
145 | | }
    | |_^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#derivable_impls
    = note: `-D clippy::derivable-impls` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::derivable_impls)]`
help: replace the manual implementation with a derive attribute and mark the default variant
    |
131 + #[derive(Default)]
132 | pub enum PrecisionPolicy {
133 |     /// FP32 only — maximum accuracy, baseline performance.
134 |     Fp32,
135 |     /// FP16 mixed precision — 2× throughput on Tensor Cores.
136 ~     #[default]
137 ~     Fp16,
    |

error: you should consider adding a `Default` implementation for `InferenceMetrics`
   --> crates/rave-tensorrt/src/tensorrt.rs:182:5
    |
182 | /     pub const fn new() -> Self {
183 | |         Self {
184 | |             frames_inferred: AtomicU64::new(0),
185 | |             total_inference_us: AtomicU64::new(0),
...   |
188 | |     }
    | |_____^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#new_without_default
    = note: `-D clippy::new-without-default` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::new_without_default)]`
help: try adding this
    |
181 + impl Default for InferenceMetrics {
182 +     fn default() -> Self {
183 +         Self::new()
184 +     }
185 + }
    |

error: you should consider adding a `Default` implementation for `RingMetrics`
   --> crates/rave-tensorrt/src/tensorrt.rs:232:5
    |
232 | /     pub const fn new() -> Self {
233 | |         Self {
234 | |             slot_reuse_count: AtomicU64::new(0),
235 | |             slot_contention_events: AtomicU64::new(0),
...   |
238 | |     }
    | |_____^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#new_without_default
help: try adding this
    |
231 + impl Default for RingMetrics {
232 +     fn default() -> Self {
233 +         Self::new()
234 +     }
235 + }
    |

error: casting raw pointers to the same type and constness is unnecessary (`*mut std::ffi::c_void` -> `*mut std::ffi::c_void`)
   --> crates/rave-nvcodec/src/nvenc.rs:268:30
    |
268 |         open_params.device = target_cuda_ctx as *mut c_void;
    |                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `target_cuda_ctx`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast
    = note: `-D clippy::unnecessary-cast` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::unnecessary_cast)]`

error: casting raw pointers to the same type and constness is unnecessary (`*mut std::ffi::c_void` -> `*mut std::ffi::c_void`)
   --> crates/rave-nvcodec/src/nvenc.rs:290:41
    |
290 |             current_cuda_ctx = %ptr_hex(current_ctx as *mut c_void),
    |                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `current_ctx`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast

error: casting raw pointers to the same type and constness is unnecessary (`*mut std::ffi::c_void` -> `*mut std::ffi::c_void`)
   --> crates/rave-nvcodec/src/nvenc.rs:291:40
    |
291 |             target_cuda_ctx = %ptr_hex(target_cuda_ctx as *mut c_void),
    |                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `target_cuda_ctx`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast

error: struct `OutputRing` has a public `len` method, but no `is_empty` method
   --> crates/rave-tensorrt/src/tensorrt.rs:391:5
    |
391 |     pub fn len(&self) -> usize {
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#len_without_is_empty
    = note: `-D clippy::len-without-is-empty` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::len_without_is_empty)]`

error: this `if` statement can be collapsed
   --> crates/rave-nvcodec/src/nvenc.rs:359:9
    |
359 | /         if !got_preset {
360 | |             if let Some(get_preset_legacy_fn) = get_preset_fn {
361 | |                 for ver in version_candidates {
362 | |                     preset_config = unsafe { std::mem::zeroed() };
...   |
386 | |         }
    | |_________^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#collapsible_if
    = note: `-D clippy::collapsible-if` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::collapsible_if)]`
help: collapse nested if block
    |
359 ~         if !got_preset
360 ~             && let Some(get_preset_legacy_fn) = get_preset_fn {
361 |                 for ver in version_candidates {
...
384 |                 }
385 ~             }
    |

error: this `if` statement can be collapsed
   --> crates/rave-tensorrt/src/tensorrt.rs:714:13
    |
714 | /             if let Ok(exe) = env::current_exe() {
715 | |                 if let Some(dir) = exe.parent() {
716 | |                     dirs.push((dir.to_path_buf(), "exe_dir"));
717 | |                     dirs.push((dir.join("deps"), "exe_dir/deps"));
718 | |                 }
719 | |             }
    | |_____________^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#collapsible_if
    = note: `-D clippy::collapsible-if` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::collapsible_if)]`
help: collapse nested if block
    |
714 ~             if let Ok(exe) = env::current_exe()
715 ~                 && let Some(dir) = exe.parent() {
716 |                     dirs.push((dir.to_path_buf(), "exe_dir"));
717 |                     dirs.push((dir.join("deps"), "exe_dir/deps"));
718 ~                 }
    |

error: this `if` statement can be collapsed
   --> crates/rave-tensorrt/src/tensorrt.rs:721:13
    |
721 | /             if let Ok(exe) = env::current_exe() {
722 | |                 if let Some(dir) = exe.parent() {
723 | |                     dirs.push((dir.to_path_buf(), "exe_dir"));
724 | |                     dirs.push((dir.join("deps"), "exe_dir/deps"));
725 | |                 }
726 | |             }
    | |_____________^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#collapsible_if
help: collapse nested if block
    |
721 ~             if let Ok(exe) = env::current_exe()
722 ~                 && let Some(dir) = exe.parent() {
723 |                     dirs.push((dir.to_path_buf(), "exe_dir"));
724 |                     dirs.push((dir.join("deps"), "exe_dir/deps"));
725 ~                 }
    |

error: writing `&PathBuf` instead of `&Path` involves a new object where a slice will do
   --> crates/rave-tensorrt/src/tensorrt.rs:784:26
    |
784 |     fn dlopen_path(path: &PathBuf, flags: i32) -> Result<()> {
    |                          ^^^^^^^^ help: change this to: `&Path`
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#ptr_arg
    = note: `-D clippy::ptr-arg` implied by `-D warnings`
    = help: to override `-D warnings` add `#[allow(clippy::ptr_arg)]`

error: this `if` statement can be collapsed
   --> crates/rave-nvcodec/src/nvenc.rs:708:9
    |
708 | /         if !self.bitstream_buf.is_null() {
709 | |             if let Some(destroy_fn) = self.fns.nvEncDestroyBitstreamBuffer {
710 | |                 // SAFETY: bitstream_buf was created via nvEncCreateBitstreamBuffer.
711 | |                 unsafe {
...   |
715 | |         }
    | |_________^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#collapsible_if
help: collapse nested if block
    |
708 ~         if !self.bitstream_buf.is_null()
709 ~             && let Some(destroy_fn) = self.fns.nvEncDestroyBitstreamBuffer {
710 |                 // SAFETY: bitstream_buf was created via nvEncCreateBitstreamBuffer.
...
713 |                 }
714 ~             }
    |

error: this `if` statement can be collapsed
   --> crates/rave-nvcodec/src/nvenc.rs:718:9
    |
718 | /         if !self.encoder.is_null() {
719 | |             if let Some(destroy_fn) = self.fns.nvEncDestroyEncoder {
720 | |                 // SAFETY: encoder was opened via nvEncOpenEncodeSessionEx.
721 | |                 unsafe {
...   |
725 | |         }
    | |_________^
    |
    = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#collapsible_if
help: collapse nested if block
    |
718 ~         if !self.encoder.is_null()
719 ~             && let Some(destroy_fn) = self.fns.nvEncDestroyEncoder {
720 |                 // SAFETY: encoder was opened via nvEncOpenEncodeSessionEx.
...
723 |                 }
724 ~             }
    |

error: casting to the same type is unnecessary (`u64` -> `u64`)
    --> crates/rave-tensorrt/src/tensorrt.rs:1217:26
     |
1217 |         let output_ptr = *(*output_arc).device_ptr() as u64;
     |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ help: try: `*(*output_arc).device_ptr()`
     |
     = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#unnecessary_cast
     = note: `-D clippy::unnecessary-cast` implied by `-D warnings`
     = help: to override `-D warnings` add `#[allow(clippy::unnecessary_cast)]`

error: could not compile `rave-nvcodec` (lib test) due to 7 previous errors
error: this `if` statement can be collapsed
    --> crates/rave-tensorrt/src/tensorrt.rs:1304:9
     |
1304 | /         if let Ok(mut guard) = self.state.try_lock() {
1305 | |             if let Some(state) = guard.take() {
1306 | |                 let _ = self.ctx.sync_all();
1307 | |                 drop(state);
1308 | |             }
1309 | |         }
     | |_________^
     |
     = help: for further information visit https://rust-lang.github.io/rust-clippy/rust-1.93.0/index.html#collapsible_if
help: collapse nested if block
     |
1304 ~         if let Ok(mut guard) = self.state.try_lock()
1305 ~             && let Some(state) = guard.take() {
1306 |                 let _ = self.ctx.sync_all();
1307 |                 drop(state);
1308 ~             }
     |

error: could not compile `rave-nvcodec` (lib) due to 7 previous errors
error: could not compile `rave-tensorrt` (lib) due to 9 previous errors
error: could not compile `rave-tensorrt` (lib test) due to 9 previous errors

[exit_code]=101
```

## Command
```bash
cargo test --workspace
```

### Output
```text
    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.12s
     Running unittests src/lib.rs (target/debug/deps/rave-7db558339aae7909)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/bin/cuda_probe.rs (target/debug/deps/cuda_probe-2d0aac38333f6f5e)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/main.rs (target/debug/deps/rave-fe82bb7fc6025e09)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/cli_subcommands.rs (target/debug/deps/cli_subcommands-51c1f353dded2f35)

running 4 tests
test help_lists_subcommands ... ok
test benchmark_help_lists_skip_encode_and_json_out ... ok
test benchmark_dry_run_emits_valid_json_shape ... ok
test upscale_dry_run_accepts_subcommand_args ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s

     Running tests/wsl_healthcheck.rs (target/debug/deps/wsl_healthcheck-f42a900fe0c56870)

running 1 test
test wsl_healthcheck_script_passes_when_enabled ... ignored, WSL host healthcheck; opt-in via env

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/lib.rs (target/debug/deps/rave_core-341999a37305cf47)

running 3 tests
test context::tests::bucket_sizing_large ... ok
test context::tests::bucket_sizing_small ... ok
test context::tests::bucket_sizing_zero ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/lib.rs (target/debug/deps/rave_cuda-b6917664f26814fd)

running 3 tests
test kernels::tests::falls_back_to_cuda_home_include_when_cuda_path_missing ... ok
test kernels::tests::falls_back_to_usr_local_cuda_include ... ok
test kernels::tests::prefers_cuda_path_include_when_present ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/lib.rs (target/debug/deps/rave_ffmpeg-f35d92ce51ce8719)

running 1 test
test ffmpeg_demuxer::tests::empty_packet_data_is_safe ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/lib.rs (target/debug/deps/rave_nvcodec-d9cf2cbe175ff087)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/lib.rs (target/debug/deps/rave_pipeline-397fe2d5a8d3f640)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/lib.rs (target/debug/deps/rave_tensorrt-092f27e82b3de4f9)

running 2 tests
test tensorrt::tests::ort_registers_tensorrt_ep_smoke ... ignored, requires model + full ORT/TensorRT runtime
test tensorrt::tests::providers_load_with_bridge_preloaded ... ignored, requires ORT TensorRT provider libs on host

test result: ok. 0 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/provider_bridge_smoke.rs (target/debug/deps/provider_bridge_smoke-6ccf764da6f683a9)

running 2 tests
test ort_tensorrt_ep_registration_smoke ... ignored, requires model + full TensorRT runtime
test providers_shared_then_tensorrt_dlopen_smoke ... ignored, requires ORT TensorRT provider libs present on host

test result: ok. 0 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests rave

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests rave_core

running 1 test
test crates/rave-core/src/debug_alloc.rs - debug_alloc (line 8) ... ignored

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

all doctests ran in 0.24s; merged doctests compilation took 0.24s
   Doc-tests rave_cuda

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests rave_ffmpeg

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests rave_nvcodec

running 1 test
test crates/rave-nvcodec/src/nvdec.rs - nvdec::wait_for_event (line 624) ... ignored

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

all doctests ran in 0.25s; merged doctests compilation took 0.25s
   Doc-tests rave_pipeline

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests rave_tensorrt

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s


[exit_code]=0
```

## Command
```bash
cargo build --workspace --release
```

### Output
```text
    Finished `release` profile [optimized] target(s) in 0.07s

[exit_code]=0
```

## Summary

cargo fmt --check -> [exit_code]=0
cargo clippy --workspace --all-targets -- -D warnings -> [exit_code]=101
cargo test --workspace -> [exit_code]=0
cargo build --workspace --release -> [exit_code]=0
