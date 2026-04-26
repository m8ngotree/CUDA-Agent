"""Pydantic schema for crawled operator entries.

A single entry describes one PyTorch operator class (e.g. `torch.nn.Linear`,
`transformers.models.llama.modeling_llama.LlamaMLP`). Both the torch crawler
and the transformers crawler emit records of this shape so synthesis can
treat them uniformly.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OperatorEntry(BaseModel):
    """One crawled operator. Source-of-truth for everything downstream."""

    name: str = Field(..., description="Fully qualified name, e.g. torch.nn.Linear")
    source_lib: str = Field(..., description="One of 'torch', 'transformers'")
    class_def: str = Field(..., description="Full Python source of the class")
    init_signature: dict[str, Any] = Field(
        default_factory=dict,
        description="__init__ parameter names → default values (None when no default)",
    )
    forward_signature: dict[str, Any] = Field(
        default_factory=dict,
        description="forward parameter names → default/annotation hint",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Coarse buckets: linear, activation, normalization, conv, mlp, attention, ...",
    )

    def signature_summary(self) -> str:
        """Compact one-liner used in synthesis prompts."""
        params = ", ".join(self.init_signature.keys()) or "—"
        return f"{self.name}({params})"
