from __future__ import annotations

from copy import copy
from functools import reduce
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, cast

import astropy.units as u
import numpy as np
from pydantic import ValidationError

from opencosmo.column.column import ColumnMask, CompoundColumnMask
from opencosmo.remote.client import (
    RemoteClient,
    RemoteQueryResponse,
    get_profile,
)
from opencosmo.remote.protocol import (
    BoundOperation,
    ComparisonPredicate,
    CompoundPredicate,
    FileCollectionSource,
    FilterOperation,
    QueryValue,
    RemoteQueryRequest,
    RemoteQuerySource,
    SelectOperation,
    StructuredCatalogSource,
    SortByOperation,
    TakeOperation,
    UnitConversionSpec,
    WithUnitsOperation,
)

if TYPE_CHECKING:
    from opencosmo.remote.client import RemoteProfile
    from opencosmo.remote.protocol import (
        Predicate,
        RemoteOperation,
        RemoteQueryProduct,
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
    ):
        self.__source = source
        self.__operations = operations

    @property
    def source(self) -> RemoteQuerySource:
        return self.__source

    @property
    def operations(self) -> tuple[RemoteOperation, ...]:
        return self.__operations

    def __with_operation(self, operation: RemoteOperation):
        return RemoteQuery(self.__source, self.__operations + (operation,))

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
        **columns_by_dataset: str | Iterable[str] | dict,
    ) -> RemoteQuery:
        if columns and columns_by_dataset:
            raise ValueError("Use either positional columns or columns_by_dataset.")
        if not columns and not columns_by_dataset:
            return self
        if columns:
            return self.__with_operation(
                SelectOperation(columns=_flatten_columns(columns))
            )

        selections = {}
        for dataset, selection in columns_by_dataset.items():
            if isinstance(selection, dict):
                if selection.get("derived_columns"):
                    raise ValueError("Remote select does not support derived columns.")
                selection = selection.get("columns", ())
            selections[dataset] = _normalize_columns(selection)
        return self.__with_operation(SelectOperation(columns_by_dataset=selections))

    def take(self, n: int, at: str = "random") -> RemoteQuery:
        at = _validate_take_position(at)
        return self.__with_operation(TakeOperation(n=n, at=at))

    def sort_by(self, column: str, invert: bool = False) -> RemoteQuery:
        return self.__with_operation(SortByOperation(column=column, invert=invert))

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
        )

    def serialize(self) -> dict:
        return self.into_request().model_dump(mode="json")

    def get(self, profile: RemoteProfile | None = None) -> RemoteQueryResponse:
        client = RemoteClient(get_profile(profile))
        accepted = client.submit(self.into_request())
        return RemoteQueryResponse(accepted.job_id, client)


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
                column=mask.column_name,
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
