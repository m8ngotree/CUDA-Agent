#!/usr/bin/env python3
"""Correctness verification for a Triton-based ModelNew.

Mirrors `utils/verification.py` but with a relaxed atol of 1e-3 (vs CUDA's
1e-5) to accommodate Triton's fp32 accumulation differences. We also keep
the same `block_torch_functional` guard, but Triton kernels go through
custom Python ops, not `torch.nn.functional`, so the guard is essentially
about catching reward-hacking attempts.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from model import Model, get_inputs, get_init_inputs
from model_new import ModelNew


VERIFICATION_ATOL = 1e-3
VERIFICATION_RTOL = 1e-3


def transform_tensors(tensors, fn):
    if isinstance(tensors, torch.Tensor):
        return fn(tensors)
    if isinstance(tensors, (list, tuple)):
        return [transform_tensors(x, fn) for x in tensors]
    if isinstance(tensors, dict):
        return {k: transform_tensors(v, fn) for k, v in tensors.items()}
    return tensors


def check_equal(actual, expected) -> None:
    assert type(actual) == type(expected), f"{type(actual)=} != {type(expected)=}"
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), f"{len(actual)=} != {len(expected)=}"
        for x, y in zip(actual, expected):
            check_equal(x, y)
    elif isinstance(actual, dict):
        for key, val in expected.items():
            assert key in actual, f"Missing key in output: {key}"
            check_equal(actual[key], val)
    elif isinstance(actual, (str, float, int)):
        assert actual == expected, f"{actual=} != {expected=}"
    elif isinstance(actual, torch.Tensor):
        torch.testing.assert_close(
            actual, expected, atol=VERIFICATION_ATOL, rtol=VERIFICATION_RTOL
        )
    else:
        raise TypeError(f"Unsupported output type: {type(actual)}")


@contextmanager
def block_torch_functional(excludes=None):
    """Same anti-hacking guard as the CUDA path: any call to
    `torch.nn.functional.*` from inside ModelNew.forward raises."""
    if excludes is None:
        excludes = set()

    originals = {}
    for name in dir(F):
        attr = getattr(F, name)
        if callable(attr) and not name.startswith("_") and name not in excludes:
            originals[name] = attr

            def wrapper(*args, __name=name, **kwargs):
                raise RuntimeError(
                    f"torch.nn.functional.{__name} is not allowed in ModelNew.forward"
                )

            setattr(F, name, wrapper)

    try:
        yield
    finally:
        for name, attr in originals.items():
            setattr(F, name, attr)


def initialize_models() -> tuple[torch.nn.Module, torch.nn.Module]:
    init_inputs = get_init_inputs()
    if not isinstance(init_inputs, (list, tuple)):
        init_inputs = [init_inputs]
    torch_model = Model(*init_inputs).eval().cuda()
    triton_model = ModelNew(*init_inputs).eval().cuda()
    triton_model.load_state_dict(torch_model.state_dict())
    return torch_model, triton_model


def build_inputs():
    inputs = get_inputs()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    inputs = transform_tensors(inputs, lambda x: x.cuda())
    return inputs, [t.clone() for t in inputs]


def main() -> None:
    torch_model, triton_model = initialize_models()
    num_checks = 5

    with torch.no_grad():
        for i in range(num_checks):
            torch_inputs, triton_inputs = build_inputs()
            torch_output = torch_model(*torch_inputs)
            with block_torch_functional():
                triton_output = triton_model(*triton_inputs)
            check_equal(triton_output, torch_output)
            print(f"[PASS] check {i + 1}/{num_checks}")

    torch.cuda.synchronize()
    print("[PASS] verify success")


if __name__ == "__main__":
    main()
