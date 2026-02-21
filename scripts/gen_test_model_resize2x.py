#!/usr/bin/env python3
"""Generate a tiny deterministic ONNX resize-2x RGB model.

The model graph is:
  input [1,3,H,W] float32
    -> Resize(scales=[1,1,2,2], nearest)
  output [1,3,2H,2W] float32

This script writes protobuf bytes directly using ONNX proto field tags so it has
no third-party Python dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path


# ONNX TensorProto::DataType FLOAT enum value.
DT_FLOAT = 1
# ONNX AttributeProto::AttributeType STRING enum value.
ATTR_STRING = 3


def _varint(value: int) -> bytes:
    value = int(value)
    if value < 0:
        raise ValueError("varint encoder expects non-negative integers")
    out = bytearray()
    while True:
        to_write = value & 0x7F
        value >>= 7
        if value:
            out.append(to_write | 0x80)
        else:
            out.append(to_write)
            return bytes(out)


def _key(field_number: int, wire_type: int) -> bytes:
    return _varint((field_number << 3) | wire_type)


def _field_varint(field_number: int, value: int) -> bytes:
    return _key(field_number, 0) + _varint(value)


def _field_len(field_number: int, payload: bytes) -> bytes:
    return _key(field_number, 2) + _varint(len(payload)) + payload


def _field_str(field_number: int, value: str) -> bytes:
    return _field_len(field_number, value.encode("utf-8"))


def _make_dimension(dim: int | str) -> bytes:
    if isinstance(dim, int):
        # TensorShapeProto.Dimension.dim_value = 1
        return _field_varint(1, dim)
    # TensorShapeProto.Dimension.dim_param = 2
    return _field_str(2, dim)


def _make_tensor_shape(dims: list[int | str]) -> bytes:
    payload = bytearray()
    for dim in dims:
        # TensorShapeProto.dim = 1 (repeated message)
        payload += _field_len(1, _make_dimension(dim))
    return bytes(payload)


def _make_type_proto_tensor_float(dims: list[int | str]) -> bytes:
    # TypeProto.Tensor.elem_type = 1
    tensor = bytearray(_field_varint(1, DT_FLOAT))
    # TypeProto.Tensor.shape = 2
    tensor += _field_len(2, _make_tensor_shape(dims))
    # TypeProto.tensor_type = 1
    return _field_len(1, bytes(tensor))


def _make_value_info(name: str, dims: list[int | str]) -> bytes:
    payload = bytearray(_field_str(1, name))  # ValueInfoProto.name
    payload += _field_len(2, _make_type_proto_tensor_float(dims))  # ValueInfoProto.type
    return bytes(payload)


def _make_tensor(name: str, dims: list[int], data_type: int, raw_data: bytes) -> bytes:
    payload = bytearray()
    for dim in dims:
        payload += _field_varint(1, dim)  # TensorProto.dims
    payload += _field_varint(2, data_type)  # TensorProto.data_type
    payload += _field_str(8, name)  # TensorProto.name
    if raw_data:
        payload += _field_len(9, raw_data)  # TensorProto.raw_data
    return bytes(payload)


def _make_attribute_string(name: str, value: str) -> bytes:
    payload = bytearray(_field_str(1, name))  # AttributeProto.name
    payload += _field_len(4, value.encode("utf-8"))  # AttributeProto.s
    payload += _field_varint(20, ATTR_STRING)  # AttributeProto.type
    return bytes(payload)


def _make_resize_node() -> bytes:
    payload = bytearray()
    for input_name in ("input", "roi", "scales"):
        payload += _field_str(1, input_name)  # NodeProto.input
    payload += _field_str(2, "output")  # NodeProto.output
    payload += _field_str(4, "Resize")  # NodeProto.op_type
    payload += _field_len(5, _make_attribute_string("mode", "nearest"))
    payload += _field_len(
        5,
        _make_attribute_string("coordinate_transformation_mode", "asymmetric"),
    )
    payload += _field_len(5, _make_attribute_string("nearest_mode", "floor"))
    return bytes(payload)


def _make_graph() -> bytes:
    payload = bytearray(_field_str(2, "rave_resize2x_rgb"))  # GraphProto.name

    node = _make_resize_node()
    payload += _field_len(1, node)  # GraphProto.node

    roi = _make_tensor("roi", [0], DT_FLOAT, b"")
    scales = _make_tensor(
        "scales",
        [4],
        DT_FLOAT,
        struct.pack("<4f", 1.0, 1.0, 2.0, 2.0),
    )
    payload += _field_len(5, roi)  # GraphProto.initializer
    payload += _field_len(5, scales)  # GraphProto.initializer

    input_info = _make_value_info("input", [1, 3, "in_h", "in_w"])
    output_info = _make_value_info("output", [1, 3, "out_h", "out_w"])
    payload += _field_len(11, input_info)  # GraphProto.input
    payload += _field_len(12, output_info)  # GraphProto.output

    return bytes(payload)


def _make_model() -> bytes:
    payload = bytearray()
    payload += _field_varint(1, 9)  # ModelProto.ir_version
    payload += _field_str(2, "rave-gen-test-model")  # ModelProto.producer_name
    payload += _field_str(3, "1.0")  # ModelProto.producer_version
    payload += _field_varint(5, 1)  # ModelProto.model_version
    payload += _field_len(7, _make_graph())  # ModelProto.graph

    # ModelProto.opset_import (default domain, opset 13)
    opset = _field_varint(2, 13)  # OperatorSetIdProto.version
    payload += _field_len(8, opset)

    return bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="tests/assets/models/resize2x_rgb.onnx",
        help="Output ONNX path",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = _make_model()
    out_path.write_bytes(model)

    digest = hashlib.sha256(model).hexdigest()
    print(f"wrote {out_path} ({len(model)} bytes, sha256={digest})")


if __name__ == "__main__":
    main()
