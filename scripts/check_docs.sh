#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "docs-lint: $*" >&2
  exit 1
}

# Known stale claims that must not reappear.
grep -q 'VramExhausted' ARCHITECTURE.md && \
  fail "ARCHITECTURE.md references non-existent EngineError::VramExhausted"

grep -q 'rave::prelude' README.md && \
  fail "README.md references non-existent rave::prelude"

grep -Fq 'When `max_batch > 1`, the inference stage collects up to `max_batch` frames before dispatching a single TensorRT invocation' ARCHITECTURE.md && \
  fail "ARCHITECTURE.md still claims active micro-batching runtime scheduling"

grep -q '"command":"benchmark|upscale"' README.md && \
  fail "README.md progress JSONL schema omits validate"

# Positive assertions to keep wording anchored to current behavior.
grep -qi 'micro-batching' ARCHITECTURE.md || \
  fail "ARCHITECTURE.md missing micro-batching section"
grep -Eiq 'single-frame|not implemented' ARCHITECTURE.md || \
  fail "ARCHITECTURE.md micro-batching section must state current non-implementation"

grep -q '"command":"benchmark|upscale|validate"' README.md || \
  fail "README.md progress JSONL schema should include validate"

echo "docs-lint: passed"
