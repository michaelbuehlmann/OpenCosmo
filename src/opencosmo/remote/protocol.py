from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


PROTOCOL_VERSION: Literal["1.0"] = "1.0"
ComparisonOperator: TypeAlias = Literal["eq", "ne", "gt", "ge", "lt", "le", "isin"]
CompoundOperator: TypeAlias = Literal["and", "or"]
TakePosition: TypeAlias = Literal["start", "end", "random"]
RemoteQueryProduct: TypeAlias = Literal["snapshot", "lightcone"]


class RemoteQuerySource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    remote_dataset: str
    product: RemoteQueryProduct
    steps: tuple[int, ...]
    catalogs: tuple[str, ...]
    open_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_source(self):
        if not self.remote_dataset:
            raise ValueError("remote_dataset must not be empty")
        if not self.steps:
            raise ValueError("steps must not be empty")
        if any(step <= 0 for step in self.steps):
            raise ValueError("steps must be positive integers")
        if not self.catalogs:
            raise ValueError("catalogs must not be empty")
        return self


class QueryValue(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: Any
    unit: str | None = None


class ComparisonPredicate(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["comparison"] = "comparison"
    column: str
    operator: ComparisonOperator
    value: QueryValue


class CompoundPredicate(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["compound"] = "compound"
    operator: CompoundOperator
    left: ComparisonPredicate | CompoundPredicate
    right: ComparisonPredicate | CompoundPredicate


Predicate: TypeAlias = Annotated[
    ComparisonPredicate | CompoundPredicate, Field(discriminator="type")
]


class FilterOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["filter"] = "filter"
    predicate: Predicate


class SelectOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["select"] = "select"
    columns: tuple[str, ...] = ()
    columns_by_dataset: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_select(self):
        if self.columns and self.columns_by_dataset:
            raise ValueError("Use either columns or columns_by_dataset, not both")
        return self


class TakeOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["take"] = "take"
    n: int
    at: TakePosition = "random"

    @model_validator(mode="after")
    def validate_take(self):
        if self.n < 0:
            raise ValueError("n must be non-negative")
        return self


class SortByOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["sort_by"] = "sort_by"
    column: str
    invert: bool = False


class UnitConversionSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversions: dict[str, str] = Field(default_factory=dict)
    columns: dict[str, str] = Field(default_factory=dict)


class WithUnitsOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["with_units"] = "with_units"
    convention: str | None = None
    conversions: dict[str, str] = Field(default_factory=dict)
    columns: dict[str, str] = Field(default_factory=dict)
    dataset_conversions: dict[str, UnitConversionSpec] = Field(default_factory=dict)


class BoundOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["bound"] = "bound"
    region: dict[str, Any]
    select_by: str | None = None


RemoteOperation: TypeAlias = Annotated[
    FilterOperation
    | SelectOperation
    | TakeOperation
    | SortByOperation
    | WithUnitsOperation
    | BoundOperation,
    Field(discriminator="type"),
]


class RemoteQueryRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    client_version: str
    source: RemoteQuerySource
    operations: tuple[RemoteOperation, ...] = ()


class RemoteQueryAccepted(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    status: Literal["queued", "running", "failed", "succeeded"]


class RemoteQueryStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    status: Literal["queued", "running", "failed", "succeeded"]
    submitted_at: str | None = None
    updated_at: str | None = None
    message: str | None = None
    result_url: str | None = None
