"""Workload requests and explicit KV migration declarations."""
from typing import Annotated, Literal
from pydantic import Field, model_validator
from astermind.common.contracts import Contract, Identifier, Nanoseconds
from .deployment import Positive

class Request(Contract):
    id: Identifier
    arrival_ns: Nanoseconds = 0
    user_device: Identifier
    input_tokens: Positive
    output_tokens: Positive = 1
    input_bytes: Nanoseconds = 0
    output_bytes_per_token: Nanoseconds = 4
    cell: Identifier | None = None
    session_id: str | None = None
    context_id: str | None = None
    parent_context_id: str | None = None
    prefix_hit_tokens: Nanoseconds = 0

    @model_validator(mode="after")
    def validate_history(self):
        if self.prefix_hit_tokens > self.input_tokens:
            raise ValueError("prefix hit exceeds prompt")
        if (self.context_id or self.parent_context_id) and not self.session_id:
            raise ValueError("context identity requires session_id")
        return self


class Migration(Contract):
    id: Identifier
    after_request: Identifier
    before_request: Identifier | None = None
    source_cell: Identifier
    target_cell: Identifier
    session_id: str
    context_id: str
    mode: Literal["copy", "move"] = "copy"


# Preserve public serialization identities across the file-only move.
Request.__module__ = 'astermind.builder.config'
Migration.__module__ = 'astermind.builder.config'
