from __future__ import annotations

import operator as op
from collections.abc import Iterable, Mapping
from copy import copy
from functools import partial, reduce
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

import astropy.units as u
import numpy as np
from pydantic import ValidationError

from opencosmo.column.column import (
    Column,
    ColumnMask,
    CompoundColumnMask,
    DerivedColumn,
    _exp10,
    _log10,
    _sqrt,
)
from opencosmo.remote.client import (
    RemoteClient,
    RemoteQueryResponse,
    RemoteQueryStatus,
    get_profile,
)
from opencosmo.remote.protocol import (
    BoundOperation,
    ComparisonPredicate,
    CompoundPredicate,
    DropOperation,
    BinaryExpr,
    ColumnRefExpr,
    FileCollectionSource,
    FilterOperation,
    QueryValue,
    RemoteExecutionOptions,
    RemoteQueryRequest,
    ScalarExpr,
    SelectionLeaf,
    SelectionTree,
    SelectOperation,
    StructuredCatalogSource,
    SortByOperation,
    TakeOperation,
    TakeRangeOperation,
    UnitConversionSpec,
    UnaryExpr,
    WithDatasetsOperation,
    WithUnitsOperation,
)

if TYPE_CHECKING:
    from opencosmo.remote.client import RemoteProfile
    from opencosmo.remote.protocol import (
        DerivedExpr,
        Predicate,
        RemoteExecutionOptions,
        RemoteOperation,
        RemoteQueryProduct,
        RemoteQuerySource,
        SelectionNode,
    )


def open(
    remote_dataset: str,
    catalogs: Iterable[str],
    *,
    product: RemoteQueryProduct,
    steps: int | Iterable[int],
) -> RemoteQuery:
    return RemoteQuery(
        StructuredCatalogSource(
            remote_dataset=remote_dataset,
            product=product,
            steps=_normalize_steps(steps),
            catalogs=tuple(catalogs),
        )
    )


def open_collection(
    remote_dataset: str, *, open_kwargs: dict[str, Any] | None = None
) -> RemoteQuery:
    return RemoteQuery(
        FileCollectionSource(
            remote_dataset=remote_dataset,
            open_kwargs=open_kwargs or {},
        )
    )


class RemoteQuery:
    def __init__(
        self,
        source: RemoteQuerySource,
        operations: tuple[RemoteOperation, ...] = (),
        execution: RemoteExecutionOptions | None = None,
    ):
        self.__source = source
        self.__operations = operations
        self.__execution = execution

    @property
    def source(self) -> RemoteQuerySource:
        return self.__source

    @property
    def operations(self) -> tuple[RemoteOperation, ...]:
        return self.__operations

    @property
    def execution(self) -> RemoteExecutionOptions | None:
        return self.__execution

    def __with_operation(self, operation: RemoteOperation):
        return RemoteQuery(
            self.__source,
            self.__operations + (operation,),
            execution=self.__execution,
        )

    def with_execution(
        self,
        *,
        allocation: str | None = None,
        priority: Literal["normal", "debug"] | None = None,
        walltime: str | None = None,
        reservation: str | None = None,
    ) -> RemoteQuery:
        execution = RemoteExecutionOptions(
            allocation=allocation,
            priority=priority,
            walltime=walltime,
            reservation=reservation,
        )
        return RemoteQuery(self.__source, self.__operations, execution=execution)

    def filter(self, *masks: ColumnMask | CompoundColumnMask) -> RemoteQuery:
        if not masks:
            return self
        predicate = reduce(
            lambda left, right: CompoundPredicate(
                operator="and", left=left, right=_predicate_from_mask(right)
            ),
            masks[1:],
            _predicate_from_mask(masks[0]),
        )
        return self.__with_operation(FilterOperation(predicate=predicate))

    def select(
        self,
        *columns: str | Iterable[str],
        **columns_or_derived: Any,
    ) -> RemoteQuery:
        if not columns and not columns_or_derived:
            raise ValueError("Remote select requires at least one selection.")
        if columns:
            _validate_remote_dataset_select_kwargs(columns_or_derived)
            return self.__with_operation(
                SelectOperation(
                    selection=SelectionLeaf(
                        columns=_flatten_columns(columns),
                        derived_columns=_serialize_derived_columns(columns_or_derived),
                    )
                )
            )
        return self.__with_operation(
            SelectOperation(selection=_selection_tree_from_input(columns_or_derived))
        )

    def drop(
        self,
        *columns: str | Iterable[str],
        **columns_by_dataset: str | Iterable[str],
    ) -> RemoteQuery:
        return self.__with_operation(_normalize_drop_operation(columns, columns_by_dataset))

    def take(self, n: int, at: str = "random") -> RemoteQuery:
        at = _validate_take_position(at)
        return self.__with_operation(TakeOperation(n=n, at=at))

    def take_range(self, start: int, end: int) -> RemoteQuery:
        return self.__with_operation(TakeRangeOperation(start=start, end=end))

    def sort_by(self, column: str, invert: bool = False) -> RemoteQuery:
        return self.__with_operation(SortByOperation(column=column, invert=invert))

    def with_datasets(self, datasets: str | Iterable[str]) -> RemoteQuery:
        return self.__with_operation(WithDatasetsOperation(datasets=_normalize_datasets(datasets)))

    def with_units(
        self,
        convention: Optional[str] = None,
        conversions: dict[u.Unit, u.Unit] = {},
        **columns_or_datasets: u.Unit | dict,
    ):
        dataset_conversions = {}
        columns = {}

        for key, value in columns_or_datasets.items():
            if isinstance(value, dict):
                dataset_conversions[key] = _unit_conversion_spec(value)
            else:
                columns[key] = str(value)

        return self.__with_operation(
            WithUnitsOperation(
                convention=convention,
                conversions=_serialize_unit_mapping(conversions),
                columns=columns,
                dataset_conversions=dataset_conversions,
            )
        )

    def bound(self, region, select_by: Optional[str] = None) -> RemoteQuery:
        model = region.into_model()
        return self.__with_operation(
            BoundOperation(region=model.model_dump(mode="json"), select_by=select_by)
        )

    def into_request(self) -> RemoteQueryRequest:
        return RemoteQueryRequest(
            client_version=_client_version(),
            source=self.__source,
            operations=self.__operations,
            execution=self.__execution,
        )

    def serialize(self) -> dict:
        return self.into_request().model_dump(mode="json", exclude_none=True)

    def submit(self, profile: RemoteProfile | None = None) -> RemoteQueryResponse:
        client = RemoteClient(get_profile(profile))
        accepted = client.submit(self.into_request())
        return RemoteQueryResponse(
            accepted.job_id,
            client,
            initial_status=RemoteQueryStatus(
                job_id=accepted.job_id,
                status=accepted.status,
            ),
        )

    def fetch(
        self,
        profile: RemoteProfile | None = None,
        *,
        show_status: bool = True,
    ):
        return self.submit(profile=profile).get_results(show_status=show_status)


def _client_version():
    try:
        return version("opencosmo")
    except PackageNotFoundError:
        return "unknown"


def _validate_take_position(at: str) -> Literal["start", "end", "random"]:
    if at not in ("start", "end", "random"):
        raise ValueError('"at" should be one of ("start", "end", "random")')
    return cast("Literal['start', 'end', 'random']", at)


def _normalize_steps(steps: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(steps, bool):
        raise TypeError("steps must be an integer or iterable of integers")
    if isinstance(steps, int):
        return (steps,)
    if isinstance(steps, str):
        raise TypeError("steps must be an integer or iterable of integers")
    return tuple(steps)


def _predicate_from_mask(mask: ColumnMask | CompoundColumnMask) -> Predicate:
    match mask:
        case ColumnMask():
            return ComparisonPredicate(
                column=mask.name,
                operator=mask.comparison_operator,
                value=_serialize_value(mask.value),
            )
        case CompoundColumnMask():
            return CompoundPredicate(
                operator=mask.compound_operator,
                left=_predicate_from_mask(mask.left),
                right=_predicate_from_mask(mask.right),
            )
    raise TypeError(f"Unsupported filter type {type(mask)}")


def _serialize_value(value) -> QueryValue:
    unit = None
    if isinstance(value, u.Quantity):
        unit = str(value.unit)
        value = value.value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    elif isinstance(value, tuple):
        value = list(value)
    return QueryValue(value=value, unit=unit)


def _flatten_columns(columns: Iterable[str | Iterable[str]]) -> tuple[str, ...]:
    output = []
    for column_group in columns:
        if isinstance(column_group, str):
            output.append(column_group)
        else:
            output.extend(column_group)
    return tuple(output)


def _normalize_columns(columns) -> tuple[str, ...]:
    if isinstance(columns, str):
        return (columns,)
    if columns is None:
        return ()
    return tuple(columns)


def _normalize_datasets(datasets: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(datasets, str):
        return (datasets,)

    normalized = []
    seen = set()
    for dataset in datasets:
        if dataset in seen:
            continue
        seen.add(dataset)
        normalized.append(dataset)
    return tuple(normalized)


def _normalize_drop_operation(
    columns: tuple[str | Iterable[str], ...],
    columns_by_dataset: Mapping[str, str | Iterable[str]],
):
    if columns and columns_by_dataset:
        raise ValueError(
            "Remote drop accepts either positional columns or dataset keyword columns."
        )
    if not columns and not columns_by_dataset:
        raise ValueError("Remote drop requires at least one column selection.")
    if columns:
        return DropOperation(columns=_flatten_columns(columns))

    normalized = {}
    for dataset, selection in columns_by_dataset.items():
        if isinstance(selection, dict):
            raise ValueError("Remote drop does not support nested dataset selections.")
        normalized[dataset] = _normalize_columns(selection)
    return DropOperation(columns_by_dataset=normalized)


def _validate_remote_dataset_select_kwargs(columns_or_derived: Mapping[str, Any]):
    for name, value in columns_or_derived.items():
        if _looks_like_dataset_selection(value):
            raise ValueError(
                f"Remote select keyword '{name}' looks like a dataset selection. "
                "When positional columns are provided, keyword arguments must be derived columns."
            )


def _looks_like_dataset_selection(value: Any) -> bool:
    if isinstance(value, (dict, str)):
        return True
    if isinstance(value, (Column, DerivedColumn)):
        return False
    if not isinstance(value, Iterable):
        return False

    try:
        values = tuple(value)
    except TypeError:
        return False
    return bool(values) and all(isinstance(item, str) for item in values)


def _selection_tree_from_input(selection: Mapping[str, Any]) -> SelectionTree:
    return SelectionTree(
        datasets={
            dataset: _selection_node_from_input(node, path=dataset)
            for dataset, node in selection.items()
        }
    )


def _selection_node_from_input(node: Any, *, path: str) -> SelectionNode:
    if isinstance(node, Mapping):
        leaf_keys = {"columns", "derived_columns"}
        if set(node).intersection(leaf_keys):
            extra_keys = set(node).difference(leaf_keys)
            if extra_keys:
                raise ValueError(
                    f"Remote select leaf '{path}' only supports 'columns' and 'derived_columns'."
                )
            columns = _normalize_columns(node.get("columns"))
            derived_columns = node.get("derived_columns", {})
            if not isinstance(derived_columns, Mapping):
                raise ValueError(
                    f"Remote select leaf '{path}.derived_columns' must be a mapping."
                )
            return SelectionLeaf(
                columns=columns,
                derived_columns=_serialize_derived_columns(
                    derived_columns, path=f"{path}.derived_columns"
                ),
            )

        return SelectionTree(
            datasets={
                dataset: _selection_node_from_input(child, path=f"{path}.{dataset}")
                for dataset, child in node.items()
            }
        )

    return SelectionLeaf(columns=_normalize_columns(node))


def _serialize_derived_columns(
    derived_columns: Mapping[str, Any],
    *,
    path: str = "derived_columns",
) -> dict[str, DerivedExpr]:
    return {
        name: _derived_expr_from_python(value, path=f"{path}.{name}")
        for name, value in derived_columns.items()
    }


def _derived_expr_from_python(expr: Any, *, path: str) -> DerivedExpr:
    match expr:
        case Column():
            return ColumnRefExpr(column=expr.name)
        case DerivedColumn():
            return _serialize_derived_column(expr, path=path)
        case int() | float():
            if isinstance(expr, bool):
                raise ValueError(
                    f"Remote derived expression '{path}' only supports int and float scalars."
                )
            return ScalarExpr(value=expr)
        case _:
            raise ValueError(
                f"Remote derived expression '{path}' must be a Column or DerivedColumn expression tree."
            )


def _serialize_derived_column(expr: DerivedColumn, *, path: str) -> DerivedExpr:
    operation = expr.operation
    binary_operator = {
        op.add: "add",
        op.sub: "sub",
        op.mul: "mul",
        op.truediv: "truediv",
        op.pow: "pow",
    }.get(operation)
    if binary_operator is not None:
        return BinaryExpr(
            operator=binary_operator,
            lhs=_derived_expr_from_python(expr.lhs, path=f"{path}.lhs"),
            rhs=_derived_expr_from_python(expr.rhs, path=f"{path}.rhs"),
        )

    if operation is _sqrt:
        return UnaryExpr(
            operator="sqrt",
            operand=_derived_expr_from_python(expr.lhs, path=f"{path}.operand"),
        )

    if isinstance(operation, partial) and operation.func is _log10:
        if operation.keywords != {"unit_container": u.DexUnit}:
            raise ValueError(
                f"Remote derived expression '{path}' only supports log10() with the default DexUnit container."
            )
        return UnaryExpr(
            operator="log10",
            operand=_derived_expr_from_python(expr.lhs, path=f"{path}.operand"),
        )

    if isinstance(operation, partial) and operation.func is _exp10:
        if operation.keywords != {"expected_unit_container": u.DexUnit}:
            raise ValueError(
                f"Remote derived expression '{path}' only supports exp10() with the default DexUnit container."
            )
        return UnaryExpr(
            operator="exp10",
            operand=_derived_expr_from_python(expr.lhs, path=f"{path}.operand"),
        )

    name = getattr(operation, "__name__", repr(operation))
    raise ValueError(
        f"Remote derived expression '{path}' uses unsupported operation {name!r}."
    )


def _serialize_unit_mapping(conversions: dict[u.Unit, u.Unit]) -> dict[str, str]:
    return {str(from_): str(to) for from_, to in conversions.items()}


def _unit_conversion_spec(data: dict) -> UnitConversionSpec:
    data = copy(data)
    conversions = data.pop("conversions", {})
    return UnitConversionSpec(
        conversions=_serialize_unit_mapping(conversions),
        columns={name: str(unit) for name, unit in data.items()},
    )


def model_to_region(region: dict):
    from opencosmo.spatial import builders
    from opencosmo.spatial.models import (
        BoxRegionModel,
        ConeRegionModel,
        HealpixRegionModel,
        SkyboxRegionModel,
    )

    if "pixels" in region:
        return builders.from_model(HealpixRegionModel.model_validate(region))
    if "center" in region and "radius" in region:
        return builders.from_model(ConeRegionModel.model_validate(region))
    if "p1" in region and "p2" in region:
        try:
            return builders.from_model(BoxRegionModel.model_validate(region))
        except ValidationError:
            return builders.make_skybox(
                **SkyboxRegionModel.model_validate(region).model_dump()
            )
    raise ValueError("Invalid serialized region")


def value_to_python(value: QueryValue):
    output = value.value
    if value.unit is not None:
        return output * u.Unit(value.unit)
    return output


def unit_mapping_to_python(conversions: dict[str, str]) -> dict[u.Unit, u.Unit]:
    return {u.Unit(from_): u.Unit(to) for from_, to in conversions.items()}
