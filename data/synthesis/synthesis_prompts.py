"""Prompts used by Stage 2 (combinatorial synthesis).

The output format matches the paper §3.1 verbatim and aligns with
KernelBench / `agent/workdir/model.py`: every problem is a self-contained
Python module that defines

    class Model(nn.Module):
        def __init__(self, *args): ...
        def forward(self, *tensors): ...

    def get_init_inputs():
        # constructor args for Model(...)

    def get_inputs():
        # tensors passed to model.forward(*inputs)

This is the contract the entire agent environment relies on
(`utils/verification.py` and `utils/profiling.py` import these three
symbols directly), so we must not deviate.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You are a PyTorch expert. You write minimal, deterministic, executable "
    "PyTorch modules in the KernelBench format. Your output is ONLY a Python "
    "module with `class Model`, `get_init_inputs()`, and `get_inputs()`. "
    "You do not include imports beyond `torch` / `torch.nn`, comments, "
    "example usage, or markdown fences."
)


FUSION_PROMPT_TEMPLATE = """\
You are a PyTorch expert. Given these operator classes, compose them into a single
fused computational layer in the KernelBench format.

Operators to compose (in order):
{op_blocks}

Write a single Python module with EXACTLY these three top-level definitions:

    class Model(nn.Module):
        def __init__(self, ...):
            super().__init__()
            # Initialise all operators above with the constructor args you receive.
            ...
        def forward(self, *tensors):
            # Pass tensors sequentially through the operators.
            ...

    def get_init_inputs():
        # Returns the list of positional args that Model(...) needs.
        # If Model takes no args, return [].
        return [...]

    def get_inputs():
        # Returns the list of tensors that model.forward(*inputs) needs.
        # Use realistic sizes (batch=4, common feature dims, etc).
        return [torch.randn(...), ...]

Hard constraints (failure to obey will cause the sample to be discarded):
1. The class MUST be named `Model` (not `FusedModel`, not anything else).
2. `get_init_inputs()` and `get_inputs()` MUST exist and MUST be callable
   with no arguments.
3. The Model + Module must be self-contained — do not reference outside symbols.
4. Do not use dropout or any stochastic op. No `torch.rand*` inside forward.
5. All parameters must be initialised deterministically by the operator
   constructors. Do not call `torch.manual_seed`.
6. forward() must accept the same number of positional tensor args as
   `get_inputs()` returns.
7. All intermediate tensors must be shape-compatible (insert reshape /
   permute as needed).
8. Use `import torch` and `import torch.nn as nn` at the module top. Do
   not import anything else.

Return ONLY the Python module source. No imports beyond `torch`/`torch.nn`,
no markdown fences, no example usage, no comments outside the class.
"""


OP_BLOCK_TEMPLATE = """\
# {idx}. {name} (from {source_lib}, tags={tags})
{class_def}
"""


@dataclass
class FusionPromptInputs:
    op_names: list[str]
    op_source_libs: list[str]
    op_tags: list[list[str]]
    op_class_defs: list[str]


def render_fusion_prompt(inputs: FusionPromptInputs) -> str:
    """Build the user prompt for `synthesize_fused_ops`."""
    blocks: list[str] = []
    for i, (name, lib, tags, src) in enumerate(
        zip(
            inputs.op_names,
            inputs.op_source_libs,
            inputs.op_tags,
            inputs.op_class_defs,
            strict=True,
        ),
        start=1,
    ):
        blocks.append(
            OP_BLOCK_TEMPLATE.format(
                idx=i,
                name=name,
                source_lib=lib,
                tags=",".join(tags),
                class_def=src.rstrip(),
            )
        )
    return FUSION_PROMPT_TEMPLATE.format(op_blocks="\n\n".join(blocks))
