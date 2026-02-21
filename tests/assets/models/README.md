# Test ONNX Models

## `resize2x_rgb.onnx`

- Purpose: tiny deterministic validate fixture model for GPU runners.
- Input tensor: `float32 [1,3,H,W]` (dynamic `H/W` via symbolic dims).
- Output tensor: `float32 [1,3,2H,2W]`.
- Graph: single `Resize` node with `scales=[1,1,2,2]`, `mode=nearest`,
  `coordinate_transformation_mode=asymmetric`, `nearest_mode=floor`.
- Opset: default domain opset `13`.
- File size: 318 bytes.
- SHA-256: `d04a86fc35bb1c17b58d4488b711f6dcea2bc5bf906237d27716c9f80fb02205`.

## Reproducibility

Regenerate with:

```bash
scripts/gen_test_model_resize2x.py --out tests/assets/models/resize2x_rgb.onnx
```

The generator writes ONNX protobuf bytes directly and has no external Python
package dependency.

## License / provenance

This model is generated from code in this repository and contains no third-party
weights, training data, or copied model artifacts.
