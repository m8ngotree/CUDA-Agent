"""Schema for filter outcomes. One `FilterResult` per synthesised problem."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FilterResult(BaseModel):
    problem_id: str
    passed: bool
    fail_reason: str | None = None
    eager_time_ms: float | None = None
    compile_time_ms: float | None = None
    decontamination_score: float | None = Field(
        None, description="Max MinHash Jaccard sim vs KernelBench (0..1); None if not checked."
    )
    # One shape per tensor returned by `get_inputs()`. Some samples have
    # multiple input tensors (matmul, attention), so this is a list of lists,
    # not a single shape vector.
    input_shapes: list[list[int]] | None = None


class FilteredProblem(BaseModel):
    """The full record we hand to the dataset stage. Combines the synthesis
    entry with the verification stats. Single denormalised row keeps
    downstream code dead simple."""

    problem_id: str
    pytorch_source: str = Field(
        ...,
        description=(
            "Full module source: `class Model` + `get_init_inputs()` + "
            "`get_inputs()` (KernelBench format)."
        ),
    )
    component_ops: list[str]
    source_libs: list[str]
    fusion_depth: int
    eager_time_ms: float
    compile_time_ms: float
    decontamination_score: float = 0.0
    input_shapes: list[list[int]]
    target: str = "cuda"
