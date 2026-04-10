from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import astropy.units as u

from opencosmo.remote.protocol import (
    BoundOperation,
    ComparisonPredicate,
    CompoundPredicate,
    FilterOperation,
    RemoteQueryRequest,
    RemoteQuerySource,
    SelectOperation,
    SortByOperation,
    TakeOperation,
    WithUnitsOperation,
)
from opencosmo.remote.query import (
    model_to_region,
    unit_mapping_to_python,
    value_to_python,
)

if TYPE_CHECKING:
    from opencosmo.remote.protocol import RemoteOperation


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
            if operation.columns_by_dataset:
                return data.select(**operation.columns_by_dataset)
            return data.select(operation.columns)
        case TakeOperation():
            return data.take(operation.n, at=operation.at)
        case SortByOperation():
            return data.sort_by(operation.column, invert=operation.invert)
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
