from __future__ import annotations

import operator as op
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, cast

import astropy.units as u

from opencosmo.remote.protocol import (
    BinaryExpr,
    BoundOperation,
    ColumnRefExpr,
    ComparisonPredicate,
    CompoundPredicate,
    DropOperation,
    FilterOperation,
    RemoteQueryRequest,
    RemoteQuerySource,
    ScalarExpr,
    SelectionLeaf,
    SelectionTree,
    SelectOperation,
    SortByOperation,
    TakeOperation,
    TakeRangeOperation,
    UnaryExpr,
    WithDatasetsOperation,
    WithUnitsOperation,
)
from opencosmo.remote.query import (
    model_to_region,
    unit_mapping_to_python,
    value_to_python,
)

if TYPE_CHECKING:
    from opencosmo.remote.protocol import RemoteOperation, SelectionNode


DatasetResolver = Callable[[RemoteQuerySource], Iterable[str | Path]]


def execute_remote_query(
    request: RemoteQueryRequest | dict | str,
    resolver: DatasetResolver,
    output_path: str | Path,
):
    if isinstance(request, str):
        request = RemoteQueryRequest.model_validate_json(request)
    elif isinstance(request, dict):
        request = RemoteQueryRequest.model_validate(request)

    source = request.source
    paths = tuple(resolver(source))
    import opencosmo as oc

    result = oc.open(*paths, **source.open_kwargs)
    for operation in request.operations:
        result = replay_operation(result, operation)

    oc.write(Path(output_path), result)
    return Path(output_path)


def replay_operation(data, operation: RemoteOperation):
    match operation:
        case FilterOperation():
            return data.filter(_predicate_to_mask(operation.predicate))
        case SelectOperation():
            return _replay_select(data, operation.selection)
        case DropOperation():
            if operation.columns_by_dataset:
                if not _supports_dataset_mapping_drop(data):
                    raise TypeError(
                        "Remote dataset-mapped drop is only supported for structure collections."
                    )
                return data.drop(**operation.columns_by_dataset)
            return data.drop(*operation.columns)
        case TakeOperation():
            return data.take(operation.n, at=operation.at)
        case TakeRangeOperation():
            if not hasattr(data, "take_range"):
                raise TypeError(
                    f"{type(data).__name__} does not support remote take_range()."
                )
            return data.take_range(operation.start, operation.end)
        case SortByOperation():
            return data.sort_by(operation.column, invert=operation.invert)
        case WithDatasetsOperation():
            if not hasattr(data, "with_datasets"):
                raise TypeError(
                    f"{type(data).__name__} does not support remote with_datasets()."
                )
            return data.with_datasets(operation.datasets)
        case WithUnitsOperation():
            conversions = unit_mapping_to_python(operation.conversions)
            if operation.dataset_conversions:
                dataset_conversions = {
                    name: {
                        "conversions": unit_mapping_to_python(spec.conversions),
                        **{col: u.Unit(unit) for col, unit in spec.columns.items()},
                    }
                    for name, spec in operation.dataset_conversions.items()
                }
                return data.with_units(
                    operation.convention,
                    conversions=conversions,
                    **dataset_conversions,
                )
            columns = {name: u.Unit(unit) for name, unit in operation.columns.items()}
            return data.with_units(
                operation.convention, conversions=conversions, **columns
            )
        case BoundOperation():
            return data.bound(model_to_region(operation.region), operation.select_by)
    raise TypeError(f"Unsupported remote operation {type(operation)}")


def _replay_select(data, selection: SelectionNode):
    match selection:
        case SelectionLeaf():
            derived_columns = {
                name: _derived_expr_to_python(expr)
                for name, expr in selection.derived_columns.items()
            }
            return data.select(*selection.columns, **derived_columns)
        case SelectionTree():
            if not _supports_nested_select(data):
                raise TypeError(
                    "Remote nested select is only supported for structure collections."
                )
            return data.select(**_selection_tree_to_python(selection))
    raise TypeError(f"Unsupported selection node {type(selection)}")


def _selection_tree_to_python(selection: SelectionTree) -> dict[str, object]:
    return {
        dataset: _selection_node_to_python(node)
        for dataset, node in selection.datasets.items()
    }


def _selection_node_to_python(node: SelectionNode):
    match node:
        case SelectionLeaf():
            derived_columns = {
                name: _derived_expr_to_python(expr)
                for name, expr in node.derived_columns.items()
            }
            if not derived_columns:
                return node.columns
            return {
                "columns": list(node.columns),
                "derived_columns": derived_columns,
            }
        case SelectionTree():
            return _selection_tree_to_python(node)
    raise TypeError(f"Unsupported selection node {type(node)}")


def _supports_nested_select(data) -> bool:
    import opencosmo as oc

    return isinstance(data, oc.StructureCollection)


def _supports_dataset_mapping_drop(data) -> bool:
    import opencosmo as oc

    return isinstance(data, oc.StructureCollection)


def _derived_expr_to_python(expr: ColumnRefExpr | ScalarExpr | BinaryExpr | UnaryExpr):
    import opencosmo as oc
    from opencosmo.column.column import DerivedColumn, _exp10, _log10, _sqrt

    match expr:
        case ColumnRefExpr():
            return oc.col(expr.column)
        case ScalarExpr():
            return expr.value
        case BinaryExpr():
            binary_operation = {
                "add": op.add,
                "sub": op.sub,
                "mul": op.mul,
                "truediv": op.truediv,
                "pow": op.pow,
            }[expr.operator]
            return DerivedColumn(
                _derived_expr_to_python(expr.lhs),
                _derived_expr_to_python(expr.rhs),
                binary_operation,
            )
        case UnaryExpr():
            unary_operation = cast(
                "Callable[..., object]",
                {
                    "sqrt": _sqrt,
                    "log10": partial(_log10, unit_container=u.DexUnit),
                    "exp10": partial(_exp10, expected_unit_container=u.DexUnit),
                }[expr.operator],
            )
            return DerivedColumn(
                _derived_expr_to_python(expr.operand),
                None,
                unary_operation,
            )
    raise TypeError(f"Unsupported derived expression {type(expr)}")


def _predicate_to_mask(predicate: ComparisonPredicate | CompoundPredicate):
    import opencosmo as oc

    match predicate:
        case ComparisonPredicate():
            col = oc.col(predicate.column)
            value = value_to_python(predicate.value)
            match predicate.operator:
                case "eq":
                    return col == value
                case "ne":
                    return col != value
                case "gt":
                    return col > value
                case "ge":
                    return col >= value
                case "lt":
                    return col < value
                case "le":
                    return col <= value
                case "isin":
                    return col.isin(value)
        case CompoundPredicate():
            left = _predicate_to_mask(predicate.left)
            right = _predicate_to_mask(predicate.right)
            if predicate.operator == "and":
                return left & right
            return left | right
    raise TypeError(f"Unsupported predicate {type(predicate)}")
