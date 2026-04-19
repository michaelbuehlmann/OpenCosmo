from __future__ import annotations

import re
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PROTOCOL_VERSION: Literal["3.0"] = "3.0"
ComparisonOperator: TypeAlias = Literal["eq", "ne", "gt", "ge", "lt", "le", "isin"]
CompoundOperator: TypeAlias = Literal["and", "or"]
TakePosition: TypeAlias = Literal["start", "end", "random"]
RemoteQueryProduct: TypeAlias = Literal["snapshot", "lightcone"]


class StructuredCatalogSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["structured_catalog"] = "structured_catalog"
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


class FileCollectionSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["file_collection"] = "file_collection"
    remote_dataset: str
    open_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_source(self):
        if not self.remote_dataset:
            raise ValueError("remote_dataset must not be empty")
        return self


RemoteQuerySource: TypeAlias = Annotated[
    StructuredCatalogSource | FileCollectionSource, Field(discriminator="kind")
]


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


class ColumnRefExpr(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["column_ref"] = "column_ref"
    column: str


class ScalarExpr(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["scalar"] = "scalar"
    value: int | float

    @field_validator("value")
    @classmethod
    def validate_scalar(cls, value: int | float):
        if isinstance(value, bool):
            raise ValueError("scalar values must be int or float")
        return value


class BinaryExpr(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["binary"] = "binary"
    operator: Literal["add", "sub", "mul", "truediv", "pow"]
    lhs: "DerivedExpr"
    rhs: "DerivedExpr"


class UnaryExpr(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["unary"] = "unary"
    operator: Literal["sqrt", "log10", "exp10"]
    operand: "DerivedExpr"


DerivedExpr: TypeAlias = Annotated[
    ColumnRefExpr | ScalarExpr | BinaryExpr | UnaryExpr,
    Field(discriminator="type"),
]


class SelectionLeaf(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["leaf"] = "leaf"
    columns: tuple[str, ...] = ()
    derived_columns: dict[str, DerivedExpr] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_leaf(self):
        if not self.columns and not self.derived_columns:
            raise ValueError("SelectionLeaf must not be empty")
        return self


class SelectionTree(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["tree"] = "tree"
    datasets: dict[str, "SelectionNode"] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tree(self):
        if not self.datasets:
            raise ValueError("SelectionTree must not be empty")
        return self


SelectionNode: TypeAlias = Annotated[
    SelectionLeaf | SelectionTree, Field(discriminator="type")
]


class SelectOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["select"] = "select"
    selection: SelectionNode


class DropOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["drop"] = "drop"
    columns: tuple[str, ...] = ()
    columns_by_dataset: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_drop(self):
        if bool(self.columns) == bool(self.columns_by_dataset):
            raise ValueError("Use either columns or columns_by_dataset")
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


class TakeRangeOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["take_range"] = "take_range"
    start: int
    end: int

    @model_validator(mode="after")
    def validate_take_range(self):
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class SortByOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["sort_by"] = "sort_by"
    column: str
    invert: bool = False


class WithDatasetsOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["with_datasets"] = "with_datasets"
    datasets: tuple[str, ...]

    @field_validator("datasets", mode="before")
    @classmethod
    def normalize_datasets(cls, value):
        if isinstance(value, str):
            value = (value,)
        normalized = []
        seen = set()
        for dataset in value:
            if dataset in seen:
                continue
            seen.add(dataset)
            normalized.append(dataset)
        return tuple(normalized)

    @model_validator(mode="after")
    def validate_datasets(self):
        if not self.datasets:
            raise ValueError("datasets must not be empty")
        return self


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
    | DropOperation
    | TakeOperation
    | TakeRangeOperation
    | SortByOperation
    | WithDatasetsOperation
    | WithUnitsOperation
    | BoundOperation,
    Field(discriminator="type"),
]


_WALLTIME_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")


class RemoteExecutionOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allocation: str | None = None
    node_count: int | None = None
    ranks_per_node: int | None = None
    priority: Literal["normal", "debug"] | None = None
    walltime: str | None = None
    reservation: str | None = None

    @field_validator("allocation", "reservation")
    @classmethod
    def validate_optional_non_empty_string(cls, value: str | None):
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("walltime")
    @classmethod
    def validate_walltime(cls, value: str | None):
        if value is None:
            return value
        if not _WALLTIME_PATTERN.fullmatch(value):
            raise ValueError("walltime must match HH:MM:SS")
        return value

    @field_validator("node_count", "ranks_per_node")
    @classmethod
    def validate_positive_int(cls, value: int | None, info):
        if value is None:
            return value
        if value < 1:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return value

    @model_validator(mode="after")
    def validate_has_override(self):
        if (
            self.allocation is None
            and self.node_count is None
            and self.ranks_per_node is None
            and self.priority is None
            and self.walltime is None
            and self.reservation is None
        ):
            raise ValueError("RemoteExecutionOptions must include at least one override")
        return self


class RemoteQueryRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol_version: Literal["3.0"] = PROTOCOL_VERSION
    client_version: str
    source: RemoteQuerySource
    operations: tuple[RemoteOperation, ...] = ()
    execution: RemoteExecutionOptions | None = None


class RemoteQueryAccepted(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    status: Literal["queued", "running", "failed", "succeeded"]
    facility_id: str | None = None
    facility_display_name: str | None = None
    resource_id: str | None = None
    resource_display_name: str | None = None
    facility_job_id: str | None = None


class RemoteQueryStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    status: Literal["queued", "running", "failed", "succeeded"]
    facility_id: str | None = None
    facility_display_name: str | None = None
    resource_id: str | None = None
    resource_display_name: str | None = None
    facility_job_id: str | None = None
    submitted_at: str | None = None
    updated_at: str | None = None
    message: str | None = None
    result_url: str | None = None
    failure_stage: (
        Literal["submission", "staging", "execution", "status_refresh", "manifest"]
        | None
    ) = None
    error_type: str | None = None
    error_detail: str | None = None
    stderr_excerpt: str | None = None
    stdout_excerpt: str | None = None


SelectionTree.model_rebuild()
BinaryExpr.model_rebuild()
UnaryExpr.model_rebuild()
SelectOperation.model_rebuild()
