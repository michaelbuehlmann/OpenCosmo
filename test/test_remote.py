from __future__ import annotations

import io
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError

import astropy.units as u
import numpy as np
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

import opencosmo as oc
from opencosmo.remote import _status_display as remote_status_display
from opencosmo.analysis.cli import cli
from opencosmo.remote._auth_store import (
    DEFAULT_AUTH_CLIENT_ID,
    StoredRemoteAuth,
    load_store,
    put_entry,
)
from opencosmo.remote.client import (
    DEFAULT_REMOTE_BASE_URL,
    RemoteAuthorizationRequired,
    RemoteClient,
    RemoteError,
    RemoteJobFailed,
    RemoteQueryResponse,
)
from opencosmo.remote.execution import execute_remote_query, replay_operation
from opencosmo.remote.protocol import (
    FileCollectionSource,
    RemoteQueryAccepted,
    RemoteQueryRequest,
    RemoteQueryStatus,
    StructuredCatalogSource,
)
from opencosmo.column.column import DerivedColumn


def _halo_paths(snapshot_path: Path) -> list[Path]:
    files = ["haloproperties.hdf5", "haloparticles.hdf5", "sodproperties.hdf5"]
    return [snapshot_path / file for file in files]


def _galaxy_paths(snapshot_path: Path) -> list[Path]:
    files = ["galaxyproperties.hdf5", "galaxyparticles.hdf5"]
    return [snapshot_path / file for file in files]


def _execute_and_open(query, resolver_paths: list[Path] | tuple[Path, ...], output_path: Path):
    execute_remote_query(query.into_request(), lambda _source: resolver_paths, output_path)
    return oc.open(output_path)


def _require_test_paths(*paths: Path) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        pytest.skip(f"test data not available: {missing[0]}")


def test_remote_query_serializes_dataset_operations():
    query = (
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        )
        .filter((oc.col("fof_halo_mass") > 1e13) & (oc.col("sod_halo_cdelta") < 10))
        .take(1000, at="random")
        .select("fof_halo_mass", "sod_halo_cdelta")
    )

    request = RemoteQueryRequest.model_validate(query.serialize())
    payload = query.serialize()

    assert isinstance(request.source, StructuredCatalogSource)
    assert request.source.kind == "structured_catalog"
    assert request.source.remote_dataset == "Frontier-E"
    assert request.source.product == "snapshot"
    assert request.source.steps == (205,)
    assert request.source.catalogs == ("halo_properties",)
    assert [operation.type for operation in request.operations] == [
        "filter",
        "take",
        "select",
    ]
    assert request.operations[0].predicate.type == "compound"
    assert payload["operations"][2] == {
        "type": "select",
        "selection": {
            "type": "leaf",
            "columns": ["fof_halo_mass", "sod_halo_cdelta"],
            "derived_columns": {},
        },
    }
    assert "execution" not in payload


def test_remote_query_serializes_execution_overrides():
    query = oc.remote.open(
        "Frontier-E",
        ["halo_properties"],
        product="snapshot",
        steps=205,
    ).with_execution(
        allocation="my-project",
        node_count=2,
        ranks_per_node=8,
        priority="debug",
        walltime="00:15:00",
        reservation="nightly-window",
    )

    request = RemoteQueryRequest.model_validate(query.serialize())
    payload = query.serialize()

    assert request.execution is not None
    assert request.execution.allocation == "my-project"
    assert request.execution.node_count == 2
    assert request.execution.ranks_per_node == 8
    assert request.execution.priority == "debug"
    assert request.execution.walltime == "00:15:00"
    assert request.execution.reservation == "nightly-window"
    assert payload["execution"] == {
        "allocation": "my-project",
        "node_count": 2,
        "ranks_per_node": 8,
        "priority": "debug",
        "walltime": "00:15:00",
        "reservation": "nightly-window",
    }


def test_remote_query_serializes_node_count_only_override():
    query = oc.remote.open(
        "Frontier-E",
        ["halo_properties"],
        product="snapshot",
        steps=205,
    ).with_execution(node_count=2)

    request = RemoteQueryRequest.model_validate(query.serialize())

    assert request.execution is not None
    assert request.execution.node_count == 2
    assert request.execution.ranks_per_node is None
    assert query.serialize()["execution"] == {"node_count": 2}


def test_remote_query_with_execution_returns_new_query():
    query = oc.remote.open(
        "Frontier-E",
        ["halo_properties"],
        product="snapshot",
        steps=205,
    )

    updated = query.with_execution(priority="debug")

    assert query.execution is None
    assert updated.execution is not None
    assert updated.execution.priority == "debug"


def test_remote_query_serializes_dataset_select_with_derived_columns():
    payload = (
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        )
        .select(
            "fof_halo_mass",
            fof_halo_px=oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx"),
        )
        .serialize()
    )

    assert payload["operations"] == [
        {
            "type": "select",
            "selection": {
                "type": "leaf",
                "columns": ["fof_halo_mass"],
                "derived_columns": {
                    "fof_halo_px": {
                        "type": "binary",
                        "operator": "mul",
                        "lhs": {"type": "column_ref", "column": "fof_halo_mass"},
                        "rhs": {"type": "column_ref", "column": "fof_halo_com_vx"},
                    }
                },
            },
        }
    ]


def test_remote_query_construction_does_not_submit():
    query = oc.remote.open(
        "Frontier-E",
        ["halo_properties"],
        product="snapshot",
        steps=205,
    ).select("fof_halo_mass")

    assert query.source.remote_dataset == "Frontier-E"
    assert [operation.type for operation in query.operations] == ["select"]


def test_remote_query_get_method_was_removed():
    query = oc.remote.open(
        "Frontier-E",
        ["halo_properties"],
        product="snapshot",
        steps=205,
    )

    assert not hasattr(query, "get")


def test_remote_query_request_rejects_structured_source_without_kind():
    with pytest.raises(ValidationError, match="Unable to extract tag using discriminator"):
        RemoteQueryRequest.model_validate(
            {
                "protocol_version": "3.0",
                "client_version": "test-client",
                "source": {
                    "remote_dataset": "Frontier-E",
                    "product": "snapshot",
                    "steps": [205],
                    "catalogs": ["halo_properties"],
                    "open_kwargs": {"synth_cores": True},
                },
                "operations": [],
            }
        )


def test_remote_query_request_accepts_explicit_structured_source_kind():
    request = RemoteQueryRequest.model_validate(
        {
            "protocol_version": "3.0",
            "client_version": "test-client",
            "source": {
                "kind": "structured_catalog",
                "remote_dataset": "Frontier-E",
                "product": "snapshot",
                "steps": [205],
                "catalogs": ["halo_properties"],
            },
            "operations": [],
        }
    )

    assert isinstance(request.source, StructuredCatalogSource)
    assert request.source.kind == "structured_catalog"
    assert request.source.steps == (205,)
    assert request.source.catalogs == ("halo_properties",)


def test_remote_query_protocol_models_accept_public_routing_metadata():
    accepted = RemoteQueryAccepted.model_validate(
        {
            "job_id": "job-1",
            "status": "queued",
            "facility_id": "alcf",
            "facility_display_name": "ALCF",
            "resource_id": "polaris",
            "resource_display_name": "Polaris",
            "facility_job_id": "12345",
        }
    )
    status = RemoteQueryStatus.model_validate(
        {
            "job_id": "job-1",
            "status": "running",
            "facility_id": "alcf",
            "facility_display_name": "ALCF",
            "resource_id": "polaris",
            "resource_display_name": "Polaris",
            "facility_job_id": "12345",
        }
    )

    assert accepted.facility_display_name == "ALCF"
    assert accepted.resource_display_name == "Polaris"
    assert status.facility_id == "alcf"
    assert status.resource_id == "polaris"


def test_remote_query_request_accepts_file_collection_source_kind():
    request = RemoteQueryRequest.model_validate(
        {
            "protocol_version": "3.0",
            "client_version": "test-client",
            "source": {
                "kind": "file_collection",
                "remote_dataset": "LastJourney-Diffsky-COSMOS-2026-02-17",
                "open_kwargs": {"synth_cores": True},
            },
            "operations": [],
        }
    )

    assert isinstance(request.source, FileCollectionSource)
    assert request.source.kind == "file_collection"
    assert request.source.remote_dataset == "LastJourney-Diffsky-COSMOS-2026-02-17"
    assert request.source.open_kwargs == {"synth_cores": True}


def test_remote_query_request_accepts_execution_block():
    request = RemoteQueryRequest.model_validate(
        {
            "protocol_version": "3.0",
            "client_version": "test-client",
            "source": {
                "kind": "structured_catalog",
                "remote_dataset": "Frontier-E",
                "product": "snapshot",
                "steps": [205],
                "catalogs": ["halo_properties"],
            },
            "operations": [],
            "execution": {
                "allocation": "my-project",
                "node_count": 2,
                "ranks_per_node": 16,
                "priority": "normal",
                "walltime": "01:30:00",
                "reservation": "window-7",
            },
        }
    )

    assert request.execution is not None
    assert request.execution.allocation == "my-project"
    assert request.execution.node_count == 2
    assert request.execution.ranks_per_node == 16
    assert request.execution.priority == "normal"
    assert request.execution.walltime == "01:30:00"
    assert request.execution.reservation == "window-7"


def test_remote_query_request_rejects_legacy_execution_mpi_ranks():
    with pytest.raises(ValidationError, match="mpi_ranks"):
        RemoteQueryRequest.model_validate(
            {
                "protocol_version": "3.0",
                "client_version": "test-client",
                "source": {
                    "kind": "structured_catalog",
                    "remote_dataset": "Frontier-E",
                    "product": "snapshot",
                    "steps": [205],
                    "catalogs": ["halo_properties"],
                },
                "operations": [],
                "execution": {"mpi_ranks": 16},
            }
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"allocation": "   "}, "value must not be blank"),
        ({"node_count": 0}, "node_count must be a positive integer"),
        (
            {"ranks_per_node": 0},
            "ranks_per_node must be a positive integer",
        ),
        ({"reservation": "   "}, "value must not be blank"),
        ({"walltime": "15:00"}, "walltime must match HH:MM:SS"),
        (
            {"priority": "urgent"},
            "Input should be 'normal' or 'debug'",
        ),
    ],
)
def test_remote_query_with_execution_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        ).with_execution(**kwargs)


def test_remote_query_with_execution_requires_at_least_one_override():
    with pytest.raises(ValidationError, match="include at least one override"):
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        ).with_execution()


def test_remote_query_serializes_structure_collection_select_and_units():
    query = (
        oc.remote.open(
            "Frontier-E",
            ["halo_properties", "halo_particles"],
            product="snapshot",
            steps=205,
        )
        .filter(oc.col("fof_halo_mass") > 1e13)
        .select(
            halo_properties=["fof_halo_mass", "sod_halo_cdelta"],
            dm_particles=["x", "y", "z"],
        )
        .with_units(
            "physical",
            halo_properties={"fof_halo_mass": u.kg},
            dm_particles={"conversions": {u.Mpc: u.km}, "x": u.m},
        )
    )

    request = RemoteQueryRequest.model_validate(query.serialize())
    payload = query.serialize()

    select = request.operations[1]
    assert payload["operations"][1] == {
        "type": "select",
        "selection": {
            "type": "tree",
            "datasets": {
                "halo_properties": {
                    "type": "leaf",
                    "columns": ["fof_halo_mass", "sod_halo_cdelta"],
                    "derived_columns": {},
                },
                "dm_particles": {
                    "type": "leaf",
                    "columns": ["x", "y", "z"],
                    "derived_columns": {},
                },
            },
        },
    }
    assert select.selection.type == "tree"

    units = request.operations[2]
    assert units.convention == "physical"
    assert units.dataset_conversions["halo_properties"].columns["fof_halo_mass"] == "kg"
    assert units.dataset_conversions["dm_particles"].conversions["Mpc"] == "km"


def test_remote_query_serializes_nested_structure_collection_select_with_derived_columns():
    payload = (
        oc.remote.open(
            "Frontier-E",
            ["halo_properties", "halo_particles", "galaxyproperties", "galaxyparticles"],
            product="snapshot",
            steps=205,
        )
        .select(
            halo_properties={
                "columns": ["fof_halo_mass", "fof_halo_com_vx"],
                "derived_columns": {
                    "fof_halo_px": oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
                },
            },
            dm_particles=["x", "y", "z"],
            galaxies={
                "galaxy_properties": {
                    "columns": ["gal_mass_star", "gal_com_vx"],
                    "derived_columns": {
                        "gal_star_px": oc.col("gal_mass_star") * oc.col("gal_com_vx")
                    },
                },
                "star_particles": ["x", "y", "z"],
            },
        )
        .serialize()
    )

    assert payload["operations"] == [
        {
            "type": "select",
            "selection": {
                "type": "tree",
                "datasets": {
                    "halo_properties": {
                        "type": "leaf",
                        "columns": ["fof_halo_mass", "fof_halo_com_vx"],
                        "derived_columns": {
                            "fof_halo_px": {
                                "type": "binary",
                                "operator": "mul",
                                "lhs": {
                                    "type": "column_ref",
                                    "column": "fof_halo_mass",
                                },
                                "rhs": {
                                    "type": "column_ref",
                                    "column": "fof_halo_com_vx",
                                },
                            }
                        },
                    },
                    "dm_particles": {
                        "type": "leaf",
                        "columns": ["x", "y", "z"],
                        "derived_columns": {},
                    },
                    "galaxies": {
                        "type": "tree",
                        "datasets": {
                            "galaxy_properties": {
                                "type": "leaf",
                                "columns": ["gal_mass_star", "gal_com_vx"],
                                "derived_columns": {
                                    "gal_star_px": {
                                        "type": "binary",
                                        "operator": "mul",
                                        "lhs": {
                                            "type": "column_ref",
                                            "column": "gal_mass_star",
                                        },
                                        "rhs": {
                                            "type": "column_ref",
                                            "column": "gal_com_vx",
                                        },
                                    }
                                },
                            },
                            "star_particles": {
                                "type": "leaf",
                                "columns": ["x", "y", "z"],
                                "derived_columns": {},
                            },
                        },
                    },
                },
            },
        }
    ]


def test_remote_query_serializes_drop_take_range_and_with_datasets():
    payload = (
        oc.remote.open(
            "Frontier-E",
            ["halo_properties", "halo_particles"],
            product="snapshot",
            steps=205,
        )
        .drop(halo_properties=["fof_halo_mass"], dm_particles=["x", "y"])
        .take_range(2, 8)
        .with_datasets(["halo_properties", "dm_particles", "halo_properties"])
        .serialize()
    )

    assert payload["operations"] == [
        {
            "type": "drop",
            "columns": [],
            "columns_by_dataset": {
                "halo_properties": ["fof_halo_mass"],
                "dm_particles": ["x", "y"],
            },
        },
        {"type": "take_range", "start": 2, "end": 8},
        {"type": "with_datasets", "datasets": ["halo_properties", "dm_particles"]},
    ]


def test_remote_query_serializes_dataset_drop():
    payload = (
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        )
        .drop("fof_halo_mass", "fof_halo_com_vx")
        .serialize()
    )

    assert payload["operations"] == [
        {
            "type": "drop",
            "columns": ["fof_halo_mass", "fof_halo_com_vx"],
            "columns_by_dataset": {},
        }
    ]


def test_remote_open_collection_serializes_file_collection_source():
    query = oc.remote.open_collection(
        "LastJourney-Diffsky-COSMOS-2026-02-17",
        open_kwargs={"synth_cores": True},
    ).select("ra", "dec")

    request = RemoteQueryRequest.model_validate(query.serialize())
    payload = query.serialize()

    assert isinstance(request.source, FileCollectionSource)
    assert request.source.kind == "file_collection"
    assert request.source.remote_dataset == "LastJourney-Diffsky-COSMOS-2026-02-17"
    assert request.source.open_kwargs == {"synth_cores": True}
    assert payload["source"]["kind"] == "file_collection"
    assert "steps" not in payload["source"]
    assert "catalogs" not in payload["source"]


def test_remote_query_request_rejects_client_supplied_paths():
    request = (
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        )
        .take(10)
        .serialize()
    )

    with pytest.raises(ValidationError):
        RemoteQueryRequest.model_validate(request | {"paths": ["/etc/passwd"]})

    source_with_path = request["source"] | {"path": "/etc/passwd"}
    with pytest.raises(ValidationError):
        RemoteQueryRequest.model_validate(request | {"source": source_with_path})


def test_remote_query_drop_rejects_mixed_positional_and_dataset_arguments():
    with pytest.raises(ValueError, match="either positional columns or dataset keyword"):
        (
            oc.remote.open(
                "Frontier-E",
                ["halo_properties"],
                product="snapshot",
                steps=205,
            ).drop("fof_halo_mass", halo_properties=["sod_halo_cdelta"])
        )


def test_remote_query_take_range_rejects_invalid_bounds():
    query = oc.remote.open(
        "Frontier-E",
        ["halo_properties"],
        product="snapshot",
        steps=205,
    )

    with pytest.raises(ValidationError, match="start must be non-negative"):
        query.take_range(-1, 5)

    with pytest.raises(ValidationError, match="end must be greater than or equal to start"):
        query.take_range(5, 4)


def test_remote_query_with_datasets_rejects_empty_input():
    query = oc.remote.open(
        "Frontier-E",
        ["halo_properties"],
        product="snapshot",
        steps=205,
    )

    with pytest.raises(ValidationError, match="datasets must not be empty"):
        query.with_datasets([])


def test_remote_query_select_rejects_empty_invocation():
    with pytest.raises(ValueError, match="at least one selection"):
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        ).select()


def test_remote_query_select_rejects_dataset_selection_kwargs_in_dataset_mode():
    with pytest.raises(ValueError, match="dataset selection"):
        (
            oc.remote.open(
                "Frontier-E",
                ["halo_properties"],
                product="snapshot",
                steps=205,
            ).select("fof_halo_mass", halo_properties=["sod_halo_cdelta"])
        )


def test_remote_query_select_rejects_unsupported_derived_operations():
    bad_expr = DerivedColumn(
        oc.col("fof_halo_mass"),
        oc.col("fof_halo_com_vx"),
        lambda lhs, rhs: lhs + rhs,
    )

    with pytest.raises(ValueError, match="unsupported operation"):
        (
            oc.remote.open(
                "Frontier-E",
                ["halo_properties"],
                product="snapshot",
                steps=205,
            ).select("fof_halo_mass", unsupported=bad_expr)
        )


def test_remote_query_select_rejects_non_default_log_and_exp_unit_containers():
    query = oc.remote.open(
        "Frontier-E",
        ["halo_properties"],
        product="snapshot",
        steps=205,
    )

    with pytest.raises(ValueError, match="default DexUnit container"):
        query.select(
            "fof_halo_mass",
            log_mass=oc.col("fof_halo_mass").log10(unit_container=u.MagUnit),
        )

    with pytest.raises(ValueError, match="default DexUnit container"):
        query.select(
            "fof_halo_mass",
            exp_mass=oc.col("fof_halo_mass").exp10(expected_unit_container=u.MagUnit),
        )


def test_replay_operation_dispatches_tree_select_drop_take_range_and_with_datasets(
    monkeypatch,
):
    events = []

    class FakeStructureCollection:
        def select(self, *args, **kwargs):
            events.append(("select", args, kwargs))
            return self

        def drop(self, *args, **kwargs):
            events.append(("drop", args, kwargs))
            return self

        def take_range(self, start, end):
            events.append(("take_range", start, end))
            return self

        def with_datasets(self, datasets):
            events.append(("with_datasets", tuple(datasets)))
            return self

    monkeypatch.setattr(oc, "StructureCollection", FakeStructureCollection)
    data = FakeStructureCollection()

    tree_select = (
        oc.remote.open(
            "Frontier-E",
            ["halo_properties", "halo_particles"],
            product="snapshot",
            steps=205,
        )
        .select(
            halo_properties={
                "columns": ["fof_halo_mass"],
                "derived_columns": {
                    "fof_halo_px": oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
                },
            },
            galaxies={"star_particles": ["x", "y", "z"]},
        )
        .operations[0]
    )
    drop = (
        oc.remote.open(
            "Frontier-E",
            ["halo_properties", "halo_particles"],
            product="snapshot",
            steps=205,
        )
        .drop(halo_properties=["fof_halo_mass"], dm_particles=["x"])
        .operations[0]
    )
    take_range = (
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        )
        .take_range(3, 7)
        .operations[0]
    )
    with_datasets = (
        oc.remote.open(
            "Frontier-E",
            ["halo_properties", "halo_particles"],
            product="snapshot",
            steps=205,
        )
        .with_datasets(["halo_properties", "dm_particles"])
        .operations[0]
    )

    replay_operation(data, tree_select)
    replay_operation(data, drop)
    replay_operation(data, take_range)
    replay_operation(data, with_datasets)

    assert events[0][0] == "select"
    assert events[0][2]["halo_properties"]["columns"] == ["fof_halo_mass"]
    assert "fof_halo_px" in events[0][2]["halo_properties"]["derived_columns"]
    assert events[0][2]["galaxies"] == {"star_particles": ("x", "y", "z")}
    assert events[1] == (
        "drop",
        (),
        {"halo_properties": ("fof_halo_mass",), "dm_particles": ("x",)},
    )
    assert events[2] == ("take_range", 3, 7)
    assert events[3] == ("with_datasets", ("halo_properties", "dm_particles"))


def test_replay_operation_reconstructs_derived_expression_behavior():
    captured = {}

    class FakeDataset:
        def select(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return self

    operation = (
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        )
        .select(
            "fof_halo_mass",
            px=((oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")) / 2).sqrt(),
        )
        .operations[0]
    )

    replay_operation(FakeDataset(), operation)

    assert captured["args"] == ("fof_halo_mass",)
    px = captured["kwargs"]["px"]
    values = px.evaluate(
        {
            "fof_halo_mass": np.array([18.0]),
            "fof_halo_com_vx": np.array([8.0]),
        }
    )
    assert np.allclose(values, np.array([np.sqrt(72.0)]))


def test_replay_operation_rejects_invalid_object_operation_combinations():
    class FakeDataset:
        def select(self, *args, **kwargs):
            return self

    tree_select = (
        oc.remote.open(
            "Frontier-E",
            ["halo_properties", "halo_particles"],
            product="snapshot",
            steps=205,
        )
        .select(halo_properties=["fof_halo_mass"])
        .operations[0]
    )
    with_datasets = (
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        )
        .with_datasets(["halo_properties"])
        .operations[0]
    )

    with pytest.raises(TypeError, match="nested select"):
        replay_operation(FakeDataset(), tree_select)

    with pytest.raises(TypeError, match="with_datasets"):
        replay_operation(FakeDataset(), with_datasets)


def test_remote_dataset_drop_matches_local_behavior(snapshot_path, tmp_path):
    source_path = snapshot_path / "haloproperties.hdf5"
    _require_test_paths(source_path)
    local = oc.open(source_path).drop("fof_halo_mass")
    remote = _execute_and_open(
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        ).drop("fof_halo_mass"),
        [source_path],
        tmp_path / "remote-dataset-drop.hdf5",
    )

    assert set(remote.columns) == set(local.columns)
    assert "fof_halo_mass" not in remote.columns


def test_remote_lightcone_take_range_matches_local_behavior(lightcone_path, tmp_path):
    paths = (
        lightcone_path / "step_600" / "haloproperties.hdf5",
        lightcone_path / "step_601" / "haloproperties.hdf5",
    )
    _require_test_paths(*paths)
    local = oc.open(*paths).take_range(5, 20).select("fof_halo_mass", "redshift")
    remote = _execute_and_open(
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="lightcone",
            steps=[600, 601],
        ).take_range(5, 20),
        paths,
        tmp_path / "remote-lightcone-range.hdf5",
    ).select("fof_halo_mass", "redshift")

    local_data = local.get_data()
    remote_data = remote.get_data()

    assert len(remote) == len(local)
    assert np.allclose(remote_data["fof_halo_mass"], local_data["fof_halo_mass"])
    assert np.allclose(remote_data["redshift"], local_data["redshift"])


def test_remote_structure_with_datasets_matches_local_behavior(snapshot_path, tmp_path):
    paths = _halo_paths(snapshot_path)
    _require_test_paths(*paths)
    local = oc.open(*paths).with_datasets(["halo_properties", "dm_particles"])
    remote = _execute_and_open(
        oc.remote.open(
            "Frontier-E",
            ["halo_properties", "halo_particles", "sod_properties"],
            product="snapshot",
            steps=205,
        ).with_datasets(["halo_properties", "dm_particles"]),
        paths,
        tmp_path / "remote-structure-with-datasets.hdf5",
    )

    assert set(remote.keys()) == set(local.keys()) == {"halo_properties", "dm_particles"}


def test_remote_nested_structure_select_with_derived_matches_local_behavior(
    snapshot_path, tmp_path
):
    paths = _halo_paths(snapshot_path) + _galaxy_paths(snapshot_path)
    _require_test_paths(*paths)
    local = (
        oc.open(*paths)
        .filter(oc.col("fof_halo_mass") > 1e14)
        .take(10)
        .select(
            halo_properties={
                "columns": [
                    "fof_halo_mass",
                    "fof_halo_com_vx",
                    "fof_halo_center_x",
                    "fof_halo_center_y",
                    "fof_halo_center_z",
                ],
                "derived_columns": {
                    "fof_halo_px": oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
                },
            },
            dm_particles=["x", "y", "z"],
            galaxies={
                "galaxy_properties": {
                    "columns": ["gal_mass_bar", "gal_mass_star", "gal_com_vx"],
                    "derived_columns": {
                        "gal_star_px": oc.col("gal_mass_star") * oc.col("gal_com_vx")
                    },
                },
                "star_particles": ["x", "y", "z"],
            },
        )
    )
    remote = _execute_and_open(
        oc.remote.open(
            "Frontier-E",
            [
                "halo_properties",
                "halo_particles",
                "sod_properties",
                "galaxy_properties",
                "star_particles",
            ],
            product="snapshot",
            steps=205,
        )
        .filter(oc.col("fof_halo_mass") > 1e14)
        .take(10)
        .select(
            halo_properties={
                "columns": [
                    "fof_halo_mass",
                    "fof_halo_com_vx",
                    "fof_halo_center_x",
                    "fof_halo_center_y",
                    "fof_halo_center_z",
                ],
                "derived_columns": {
                    "fof_halo_px": oc.col("fof_halo_mass") * oc.col("fof_halo_com_vx")
                },
            },
            dm_particles=["x", "y", "z"],
            galaxies={
                "galaxy_properties": {
                    "columns": ["gal_mass_bar", "gal_mass_star", "gal_com_vx"],
                    "derived_columns": {
                        "gal_star_px": oc.col("gal_mass_star") * oc.col("gal_com_vx")
                    },
                },
                "star_particles": ["x", "y", "z"],
            },
        ),
        paths,
        tmp_path / "remote-nested-select.hdf5",
    )

    local_halo = next(iter(local.halos()))
    remote_halo = next(iter(remote.halos()))

    assert set(remote_halo["halo_properties"].keys()) == set(local_halo["halo_properties"].keys())
    assert set(remote_halo["dm_particles"].columns) == set(local_halo["dm_particles"].columns)
    assert set(remote_halo["galaxies"]["galaxy_properties"].columns) == set(
        local_halo["galaxies"]["galaxy_properties"].columns
    )
    assert np.isclose(
        remote_halo["halo_properties"]["fof_halo_px"].value,
        local_halo["halo_properties"]["fof_halo_px"].value,
    )
    assert np.isclose(
        remote_halo["galaxies"]["galaxy_properties"]["gal_star_px"].value,
        local_halo["galaxies"]["galaxy_properties"]["gal_star_px"].value,
    )


def test_remote_auth_manual_token_flow(tmp_path):
    oc.remote.configure(
        oc.remote.RemoteProfile(
            base_url="https://example.test",
            auth_storage_path=tmp_path / "remote-auth.json",
        )
    )

    initial = oc.remote.auth.status()
    assert not initial.authenticated
    assert initial.token_source == "none"

    logged_in = oc.remote.auth.login(token="secret")
    assert logged_in.authenticated
    assert logged_in.token_source == "memory"
    assert oc.remote.get_profile().token == "secret"

    logged_out = oc.remote.auth.logout()
    assert not logged_out.authenticated
    assert logged_out.token_source == "none"
    assert oc.remote.get_profile().token is None


def test_remote_auth_interactive_login_stores_refreshable_auth(
    monkeypatch, tmp_path, capsys
):
    storage_path = tmp_path / "remote-auth.json"
    profile = oc.remote.RemoteProfile(
        base_url="https://example.test/",
        auth_storage_path=storage_path,
    )
    oc.remote.configure(profile)

    def urlopen(req, timeout=None, context=None):
        assert req.full_url == (
            "https://example.test/.well-known/opencosmo-remote-auth"
        )
        return _Response(
            json.dumps(
                {
                    "auth_provider": "globus",
                    "required_scope": "scope://remote",
                    "dependent_scopes": [
                        "scope://facility",
                        "scope://facility",
                        "scope://extra",
                    ],
                    "session_required_policies": ["policy-1"],
                }
            ).encode()
        )
    flow_started = []
    authorize_url_params = []

    class FakeNativeAppAuthClient:
        def __init__(self, client_id):
            assert client_id == DEFAULT_AUTH_CLIENT_ID

        def oauth2_start_flow(self, **kwargs):
            flow_started.append(kwargs)

        def oauth2_get_authorize_url(self, **kwargs):
            authorize_url_params.append(kwargs)
            return "https://auth.globus.org/authorize?scope=scope://remote"

        def oauth2_exchange_code_for_tokens(self, code):
            assert code == "auth-code"
            return _token_response(
                access_token="stored-token",
                refresh_token="refresh-token",
                scope="scope://remote scope://facility scope://extra",
                expires_at=int(time.time()) + 3600,
            )

    class FakeGlobusSDK:
        NativeAppAuthClient = FakeNativeAppAuthClient

    monkeypatch.setattr("opencosmo.remote.auth.request.urlopen", urlopen)
    monkeypatch.setattr("opencosmo.remote.auth._get_globus_sdk", lambda: FakeGlobusSDK)
    monkeypatch.setattr(
        "opencosmo.remote.auth.secrets.token_urlsafe",
        lambda _n: "pkce-verifier",
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "auth-code")

    status = oc.remote.auth.login()

    assert status.authenticated
    assert status.token_source == "stored"
    assert status.required_scope == "scope://remote"
    assert status.dependent_scopes == ("scope://facility", "scope://extra")
    assert status.session_required_policies == ("policy-1",)
    assert status.base_url == "https://example.test"
    assert status.expires_at is not None
    assert oc.remote.get_profile().token is None
    assert oc.remote.auth.status().token_source == "stored"
    assert flow_started == [
        {
            "requested_scopes": (
                "scope://remote",
                "scope://facility",
                "scope://extra",
            ),
            "redirect_uri": "https://auth.globus.org/v2/web/auth-code",
            "refresh_tokens": True,
            "verifier": "pkce-verifier",
        },
        {
            "requested_scopes": (
                "scope://remote",
                "scope://facility",
                "scope://extra",
            ),
            "redirect_uri": "https://auth.globus.org/v2/web/auth-code",
            "refresh_tokens": True,
            "verifier": "pkce-verifier",
        },
    ]
    assert authorize_url_params == [
        {
            "session_required_policies": ("policy-1",),
            "prompt": "login",
        }
    ]
    assert "authorize OpenCosmo Remote" in capsys.readouterr().out

    stored = json.loads(storage_path.read_text())
    assert stored["entries"]["https://example.test"]["access_token"] == "stored-token"
    assert stored["entries"]["https://example.test"]["refresh_token"] == "refresh-token"
    assert stored["entries"]["https://example.test"]["dependent_scopes"] == [
        "scope://facility",
        "scope://extra",
    ]
    assert stored["entries"]["https://example.test"]["session_required_policies"] == [
        "policy-1"
    ]


def test_remote_auth_start_interactive_login_returns_serializable_pending_state(
    monkeypatch, tmp_path
):
    storage_path = tmp_path / "remote-auth.json"
    profile = oc.remote.RemoteProfile(
        base_url="https://example.test/",
        auth_storage_path=storage_path,
    )
    oc.remote.configure(profile)

    def urlopen(req, timeout=None, context=None):
        assert req.full_url == (
            "https://example.test/.well-known/opencosmo-remote-auth"
        )
        return _Response(
            json.dumps(
                {
                    "auth_provider": "globus",
                    "required_scope": "scope://remote",
                    "dependent_scopes": ["scope://facility", "scope://facility"],
                    "session_required_policies": ["policy-1"],
                }
            ).encode()
        )

    observed = {"start_flow": [], "authorize_params": []}

    class FakeNativeAppAuthClient:
        def __init__(self, client_id):
            assert client_id == DEFAULT_AUTH_CLIENT_ID

        def oauth2_start_flow(self, **kwargs):
            observed["start_flow"].append(kwargs)

        def oauth2_get_authorize_url(self, **kwargs):
            observed["authorize_params"].append(kwargs)
            return "https://auth.globus.org/authorize?scope=scope://remote"

    class FakeGlobusSDK:
        NativeAppAuthClient = FakeNativeAppAuthClient

    monkeypatch.setattr("opencosmo.remote.auth.request.urlopen", urlopen)
    monkeypatch.setattr("opencosmo.remote.auth._get_globus_sdk", lambda: FakeGlobusSDK)
    monkeypatch.setattr(
        "opencosmo.remote.auth.secrets.token_urlsafe",
        lambda _n: "pkce-verifier",
    )

    pending = oc.remote.auth.start_interactive_login()

    assert pending.base_url == "https://example.test"
    assert pending.auth_client_id == DEFAULT_AUTH_CLIENT_ID
    assert pending.required_scope == "scope://remote"
    assert pending.dependent_scopes == ("scope://facility",)
    assert pending.session_required_policies == ("policy-1",)
    assert pending.requested_scopes == ("scope://remote", "scope://facility")
    assert pending.redirect_uri == "https://auth.globus.org/v2/web/auth-code"
    assert pending.verifier == "pkce-verifier"
    assert pending.authorize_url == "https://auth.globus.org/authorize?scope=scope://remote"
    assert observed["start_flow"] == [
        {
            "requested_scopes": ("scope://remote", "scope://facility"),
            "redirect_uri": "https://auth.globus.org/v2/web/auth-code",
            "refresh_tokens": True,
            "verifier": "pkce-verifier",
        }
    ]
    assert observed["authorize_params"] == [
        {
            "session_required_policies": ("policy-1",),
            "prompt": "login",
        }
    ]


def test_remote_auth_finish_interactive_login_uses_verifier_and_stores_auth(
    monkeypatch, tmp_path
):
    storage_path = tmp_path / "remote-auth.json"
    profile = oc.remote.RemoteProfile(
        base_url="https://example.test",
        auth_storage_path=storage_path,
    )
    oc.remote.configure(profile)

    flow_started = []

    class FakeNativeAppAuthClient:
        def __init__(self, client_id):
            assert client_id == DEFAULT_AUTH_CLIENT_ID

        def oauth2_start_flow(self, **kwargs):
            flow_started.append(kwargs)

        def oauth2_exchange_code_for_tokens(self, code):
            assert code == "auth-code"
            return _token_response(
                access_token="stored-token",
                refresh_token="refresh-token",
                scope="scope://remote scope://facility",
                expires_at=int(time.time()) + 3600,
            )

    class FakeGlobusSDK:
        NativeAppAuthClient = FakeNativeAppAuthClient

    monkeypatch.setattr("opencosmo.remote.auth._get_globus_sdk", lambda: FakeGlobusSDK)

    pending = oc.remote.auth.PendingInteractiveLogin(
        base_url="https://example.test",
        auth_client_id=DEFAULT_AUTH_CLIENT_ID,
        required_scope="scope://remote",
        dependent_scopes=("scope://facility",),
        requested_scopes=("scope://remote", "scope://facility"),
        redirect_uri="https://auth.globus.org/v2/web/auth-code",
        verifier="pkce-verifier",
        authorize_url="https://auth.globus.org/authorize",
    )

    status = oc.remote.auth.finish_interactive_login(pending, auth_code=" auth-code ")

    assert status.authenticated
    assert status.token_source == "stored"
    assert status.base_url == "https://example.test"
    assert status.required_scope == "scope://remote"
    assert status.dependent_scopes == ("scope://facility",)
    assert flow_started == [
        {
            "requested_scopes": ("scope://remote", "scope://facility"),
            "redirect_uri": "https://auth.globus.org/v2/web/auth-code",
            "refresh_tokens": True,
            "verifier": "pkce-verifier",
        }
    ]
    stored = json.loads(storage_path.read_text())
    assert stored["entries"]["https://example.test"]["access_token"] == "stored-token"
    assert stored["entries"]["https://example.test"]["refresh_token"] == "refresh-token"


def test_remote_auth_interactive_login_omits_prompt_without_session_policies(
    monkeypatch, tmp_path
):
    oc.remote.configure(
        oc.remote.RemoteProfile(
            base_url="https://example.test",
            auth_storage_path=tmp_path / "remote-auth.json",
        )
    )
    monkeypatch.setattr(
        "opencosmo.remote.auth.request.urlopen",
        lambda req, timeout=None, context=None: _Response(
            json.dumps(
                {
                    "auth_provider": "globus",
                    "required_scope": "scope://remote",
                    "dependent_scopes": [],
                    "session_required_policies": [],
                }
            ).encode()
        ),
    )

    authorize_url_params = []

    class FakeNativeAppAuthClient:
        def __init__(self, client_id):
            assert client_id == DEFAULT_AUTH_CLIENT_ID

        def oauth2_start_flow(self, **kwargs):
            return None

        def oauth2_get_authorize_url(self, **kwargs):
            authorize_url_params.append(kwargs)
            return "https://auth.globus.org/authorize"

        def oauth2_exchange_code_for_tokens(self, code):
            assert code == "auth-code"
            return _token_response(
                access_token="stored-token",
                refresh_token="refresh-token",
                scope="scope://remote",
                expires_at=int(time.time()) + 3600,
            )

    class FakeGlobusSDK:
        NativeAppAuthClient = FakeNativeAppAuthClient

    monkeypatch.setattr("opencosmo.remote.auth._get_globus_sdk", lambda: FakeGlobusSDK)

    oc.remote.auth.login(auth_code="auth-code")

    assert authorize_url_params == [{}]


def test_remote_auth_status_uses_persisted_auth(tmp_path):
    storage_path = tmp_path / "remote-auth.json"
    oc.remote.configure(
        oc.remote.RemoteProfile(
            base_url="https://example.test",
            auth_storage_path=storage_path,
        )
    )
    put_entry(
        storage_path,
        StoredRemoteAuth(
            base_url="https://example.test",
            client_id=DEFAULT_AUTH_CLIENT_ID,
            required_scope="scope://remote",
            access_token="stored-token",
            refresh_token="refresh-token",
            expires_at=1_800_000_000,
            dependent_scopes=("scope://facility",),
            session_required_policies=("policy-1",),
        ),
    )

    status = oc.remote.auth.status()

    assert status.authenticated
    assert status.token_source == "stored"
    assert status.required_scope == "scope://remote"
    assert status.dependent_scopes == ("scope://facility",)
    assert status.session_required_policies == ("policy-1",)
    assert status.expires_at == 1_800_000_000


def test_remote_auth_logout_clears_persisted_auth(tmp_path):
    storage_path = tmp_path / "remote-auth.json"
    oc.remote.configure(
        oc.remote.RemoteProfile(
            base_url="https://example.test",
            auth_storage_path=storage_path,
        )
    )
    put_entry(
        storage_path,
        StoredRemoteAuth(
            base_url="https://example.test",
            client_id=DEFAULT_AUTH_CLIENT_ID,
            required_scope="scope://remote",
            access_token="stored-token",
            refresh_token="refresh-token",
            expires_at=1_800_000_000,
        ),
    )

    status = oc.remote.auth.logout()

    assert not status.authenticated
    assert status.token_source == "none"
    assert not storage_path.exists()


def test_remote_auth_login_rejects_blank_required_scope(monkeypatch, tmp_path):
    oc.remote.configure(
        oc.remote.RemoteProfile(
            base_url="https://example.test",
            auth_storage_path=tmp_path / "remote-auth.json",
        )
    )
    monkeypatch.setattr(
        "opencosmo.remote.auth.request.urlopen",
        lambda req, timeout=None, context=None: _Response(
            json.dumps({"auth_provider": "globus", "required_scope": ""}).encode()
        ),
    )

    with pytest.raises(RemoteError, match="missing required_scope"):
        oc.remote.auth.login(auth_code="unused")


def test_remote_auth_login_rejects_non_globus_provider(monkeypatch, tmp_path):
    oc.remote.configure(
        oc.remote.RemoteProfile(
            base_url="https://example.test",
            auth_storage_path=tmp_path / "remote-auth.json",
        )
    )
    monkeypatch.setattr(
        "opencosmo.remote.auth.request.urlopen",
        lambda req, timeout=None, context=None: _Response(
            json.dumps(
                {"auth_provider": "oidc", "required_scope": "scope://remote"}
            ).encode()
        ),
    )

    with pytest.raises(RemoteError, match="invalid auth_provider"):
        oc.remote.auth.login(auth_code="unused")


def test_remote_auth_status_without_profile_uses_default_remote(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("opencosmo.remote.client._default_profile", None)
    monkeypatch.setattr(
        "opencosmo.remote.client.DEFAULT_AUTH_STORAGE_PATH",
        tmp_path / "remote-auth.json",
    )

    status = oc.remote.auth.status()

    assert not status.authenticated
    assert status.token_source == "none"
    assert status.message == "No remote auth token is configured."
    assert status.base_url == DEFAULT_REMOTE_BASE_URL


def test_get_profile_uses_less_aggressive_default_poll_interval(monkeypatch, tmp_path):
    monkeypatch.setattr("opencosmo.remote.client._default_profile", None)
    monkeypatch.setattr(
        "opencosmo.remote.client.DEFAULT_AUTH_STORAGE_PATH",
        tmp_path / "remote-auth.json",
    )

    profile = oc.remote.get_profile()

    assert profile.poll_interval_s == 15.0


def test_remote_auth_status_loads_legacy_store_without_new_metadata_fields(
    monkeypatch, tmp_path
):
    storage_path = tmp_path / "remote-auth.json"
    storage_path.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": {
                    "https://example.test": {
                        "client_id": DEFAULT_AUTH_CLIENT_ID,
                        "required_scope": "scope://remote",
                        "access_token": "stored-token",
                        "refresh_token": "refresh-token",
                        "expires_at": 1_800_000_000,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    oc.remote.configure(
        oc.remote.RemoteProfile(
            base_url="https://example.test",
            auth_storage_path=storage_path,
        )
    )

    status = oc.remote.auth.status()
    loaded = load_store(storage_path)["https://example.test"]

    assert status.authenticated
    assert status.dependent_scopes == ()
    assert status.session_required_policies == ()
    assert loaded.dependent_scopes == ()
    assert loaded.session_required_policies == ()


def test_remote_query_submit_returns_response_and_response_methods_still_work(
    monkeypatch, tmp_path
):
    calls = []

    class Response:
        def __init__(self, data: bytes):
            self.__data = data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return self.__data

    def urlopen(req, timeout=None, context=None):
        calls.append(req.full_url)
        if req.full_url.endswith("/queries") and req.get_method() == "POST":
            return Response(
                json.dumps({"job_id": "job-1", "status": "queued"}).encode()
            )
        if req.full_url.endswith("/queries/job-1"):
            return Response(
                RemoteQueryStatus(
                    job_id="job-1",
                    status="succeeded",
                    result_url="https://example.test/results/job-1.hdf5",
                )
                .model_dump_json()
                .encode()
            )
        return Response(b"fake-hdf5")

    def fake_open(path):
        return Path(path).read_bytes()

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)
    monkeypatch.setattr(oc, "open", fake_open)

    profile = oc.remote.RemoteProfile(
        base_url="https://example.test",
        result_cache_dir=tmp_path,
    )
    response = oc.remote.open(
        "Frontier-E",
        ["halo_properties"],
        product="snapshot",
        steps=205,
    ).submit(profile=profile)
    assert isinstance(response, RemoteQueryResponse)
    assert response.get_status().status == "succeeded"
    assert response.get_results() == b"fake-hdf5"
    assert calls == [
        "https://example.test/queries",
        "https://example.test/queries/job-1",
        "https://example.test/results/job-1.hdf5",
    ]


def test_remote_query_fetch_submits_waits_downloads_and_opens(monkeypatch, tmp_path):
    calls = []

    class Response:
        def __init__(self, data: bytes):
            self.__data = data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return self.__data

    def urlopen(req, timeout=None, context=None):
        calls.append(req.full_url)
        if req.full_url.endswith("/queries") and req.get_method() == "POST":
            return Response(
                json.dumps({"job_id": "job-1", "status": "queued"}).encode()
            )
        if req.full_url.endswith("/queries/job-1"):
            return Response(
                RemoteQueryStatus(
                    job_id="job-1",
                    status="succeeded",
                    result_url="https://example.test/results/job-1.hdf5",
                )
                .model_dump_json()
                .encode()
            )
        return Response(b"fake-hdf5")

    def fake_open(path):
        return Path(path).read_bytes()

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)
    monkeypatch.setattr(oc, "open", fake_open)

    profile = oc.remote.RemoteProfile(
        base_url="https://example.test",
        result_cache_dir=tmp_path,
    )
    result = oc.remote.open(
        "Frontier-E",
        ["halo_properties"],
        product="snapshot",
        steps=205,
    ).fetch(profile=profile)

    assert result == b"fake-hdf5"
    assert calls == [
        "https://example.test/queries",
        "https://example.test/queries/job-1",
        "https://example.test/results/job-1.hdf5",
    ]


def test_remote_query_fetch_reports_timestamped_status_transitions(
    monkeypatch, tmp_path, capsys
):
    calls = []
    statuses = iter(
        [
            RemoteQueryStatus(job_id="job-1", status="queued"),
            RemoteQueryStatus(job_id="job-1", status="queued"),
            RemoteQueryStatus(job_id="job-1", status="running"),
            RemoteQueryStatus(
                job_id="job-1",
                status="succeeded",
                result_url="https://example.test/results/job-1.hdf5",
            ),
        ]
    )
    timestamps = iter(
        [
            datetime(2026, 4, 10, 12, 0, 0),
            datetime(2026, 4, 10, 12, 0, 1),
            datetime(2026, 4, 10, 12, 0, 2),
            datetime(2026, 4, 10, 12, 0, 3),
        ]
    )

    def urlopen(req, timeout=None, context=None):
        calls.append(req.full_url)
        if req.full_url.endswith("/queries") and req.get_method() == "POST":
            return _Response(json.dumps({"job_id": "job-1", "status": "queued"}).encode())
        if req.full_url.endswith("/queries/job-1"):
            return _Response(next(statuses).model_dump_json().encode())
        return _Response(b"fake-hdf5")

    def fake_open(path):
        return Path(path).read_bytes()

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)
    monkeypatch.setattr("opencosmo.remote.client.time.sleep", lambda *_: None)
    monkeypatch.setattr("opencosmo.remote._status_display._now", lambda: next(timestamps))
    monkeypatch.setattr(oc, "open", fake_open)

    profile = oc.remote.RemoteProfile(
        base_url="https://example.test",
        poll_interval_s=0,
        result_cache_dir=tmp_path,
    )
    result = oc.remote.open(
        "Frontier-E",
        ["halo_properties"],
        product="snapshot",
        steps=205,
    ).fetch(profile=profile)

    assert result == b"fake-hdf5"
    assert capsys.readouterr().err.splitlines() == [
        "[12:00:00] Remote query job-1 submitted",
        "[12:00:01] Remote query job-1 queued (1s since submitted, 1s total)",
        "[12:00:02] Remote query job-1 running (1s since queued, 2s total)",
        "[12:00:03] Remote query job-1 succeeded (1s since running, 3s total)",
    ]
    assert calls == [
        "https://example.test/queries",
        "https://example.test/queries/job-1",
        "https://example.test/queries/job-1",
        "https://example.test/queries/job-1",
        "https://example.test/queries/job-1",
        "https://example.test/results/job-1.hdf5",
    ]


def test_remote_query_wait_uses_initial_accepted_status_once(
    monkeypatch, capsys
):
    statuses = iter(
        [
            RemoteQueryStatus(job_id="job-1", status="running"),
            RemoteQueryStatus(job_id="job-1", status="succeeded"),
        ]
    )
    timestamps = iter(
        [
            datetime(2026, 4, 10, 12, 1, 0),
            datetime(2026, 4, 10, 12, 1, 1),
            datetime(2026, 4, 10, 12, 1, 2),
        ]
    )

    def urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/queries") and req.get_method() == "POST":
            return _Response(json.dumps({"job_id": "job-1", "status": "running"}).encode())
        return _Response(next(statuses).model_dump_json().encode())

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)
    monkeypatch.setattr("opencosmo.remote.client.time.sleep", lambda *_: None)
    monkeypatch.setattr("opencosmo.remote._status_display._now", lambda: next(timestamps))

    response = oc.remote.open(
        "Frontier-E",
        ["halo_properties"],
        product="snapshot",
        steps=205,
    ).submit(profile=oc.remote.RemoteProfile(base_url="https://example.test", poll_interval_s=0))

    status = response.wait()

    assert status.status == "succeeded"
    assert capsys.readouterr().err.splitlines() == [
        "[12:01:00] Remote query job-1 submitted",
        "[12:01:01] Remote query job-1 running (1s since submitted, 1s total)",
        "[12:01:02] Remote query job-1 succeeded (1s since running, 2s total)",
    ]


def test_remote_query_get_results_can_disable_status_output(
    monkeypatch, tmp_path, capsys
):
    def urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/queries") and req.get_method() == "POST":
            return _Response(json.dumps({"job_id": "job-1", "status": "queued"}).encode())
        if req.full_url.endswith("/queries/job-1"):
            return _Response(
                RemoteQueryStatus(
                    job_id="job-1",
                    status="succeeded",
                    result_url="https://example.test/results/job-1.hdf5",
                )
                .model_dump_json()
                .encode()
            )
        return _Response(b"fake-hdf5")

    def fake_open(path):
        return Path(path).read_bytes()

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)
    monkeypatch.setattr(oc, "open", fake_open)

    response = oc.remote.open(
        "Frontier-E",
        ["halo_properties"],
        product="snapshot",
        steps=205,
    ).submit(
        profile=oc.remote.RemoteProfile(
            base_url="https://example.test",
            result_cache_dir=tmp_path,
        )
    )

    result = response.get_results(show_status=False)

    assert result == b"fake-hdf5"
    assert capsys.readouterr().err == ""


def test_remote_query_fetch_reports_failed_status_before_raising(
    monkeypatch, capsys
):
    timestamps = iter(
        [
            datetime(2026, 4, 10, 12, 2, 0),
            datetime(2026, 4, 10, 12, 2, 1),
            datetime(2026, 4, 10, 12, 2, 2),
        ]
    )

    def urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/queries") and req.get_method() == "POST":
            return _Response(json.dumps({"job_id": "job-1", "status": "queued"}).encode())
        return _Response(
            RemoteQueryStatus(
                job_id="job-1",
                status="failed",
                message="Remote query failed.",
                failure_stage="execution",
                error_type="RuntimeError",
                error_detail="out of memory",
                stderr_excerpt="stderr line 1\nstderr line 2",
                stdout_excerpt="stdout line 1",
            )
            .model_dump_json()
            .encode()
        )

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)
    monkeypatch.setattr("opencosmo.remote._status_display._now", lambda: next(timestamps))

    with pytest.raises(RemoteJobFailed) as exc_info:
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        ).fetch(profile=oc.remote.RemoteProfile(base_url="https://example.test"))

    assert str(exc_info.value) == "\n".join(
        [
            "Remote query job-1 failed.",
            "Summary: Remote query failed.",
            "Stage: execution",
            "Error type: RuntimeError",
            "Detail:",
            "  out of memory",
            "Stderr tail:",
            "  stderr line 1",
            "  stderr line 2",
            "Stdout tail:",
            "  stdout line 1",
        ]
    )
    assert capsys.readouterr().err.splitlines() == [
        "[12:02:00] Remote query job-1 submitted",
        "[12:02:01] Remote query job-1 queued (1s since submitted, 1s total)",
        "[12:02:02] Remote query job-1 failed: Remote query failed. (1s since queued, 2s total)",
    ]


def test_remote_query_failed_status_line_uses_error_detail_when_message_missing(
    monkeypatch, capsys
):
    timestamps = iter(
        [
            datetime(2026, 4, 10, 12, 3, 0),
            datetime(2026, 4, 10, 12, 3, 1),
            datetime(2026, 4, 10, 12, 3, 2),
        ]
    )

    def urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/queries") and req.get_method() == "POST":
            return _Response(json.dumps({"job_id": "job-1", "status": "queued"}).encode())
        return _Response(
            RemoteQueryStatus(
                job_id="job-1",
                status="failed",
                error_type="RuntimeError",
                error_detail="worker exploded",
            )
            .model_dump_json()
            .encode()
        )

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)
    monkeypatch.setattr("opencosmo.remote._status_display._now", lambda: next(timestamps))

    with pytest.raises(RemoteJobFailed, match="worker exploded"):
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        ).fetch(profile=oc.remote.RemoteProfile(base_url="https://example.test"))

    assert capsys.readouterr().err.splitlines() == [
        "[12:03:00] Remote query job-1 submitted",
        "[12:03:01] Remote query job-1 queued (1s since submitted, 1s total)",
        "[12:03:02] Remote query job-1 failed: RuntimeError: worker exploded (1s since queued, 2s total)",
    ]


def test_remote_query_fetch_hydrates_immediate_failed_submit_status(monkeypatch):
    def urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/queries") and req.get_method() == "POST":
            return _Response(json.dumps({"job_id": "job-1", "status": "failed"}).encode())
        return _Response(
            RemoteQueryStatus(
                job_id="job-1",
                status="failed",
                message="Remote query submission failed.",
                failure_stage="submission",
                error_type="FacilityUnauthorizedError",
                error_detail="Keycloak ID token not valid or expired. Try to re-authenticate.",
            )
            .model_dump_json()
            .encode()
        )

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)

    with pytest.raises(RemoteJobFailed) as exc_info:
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        ).fetch(profile=oc.remote.RemoteProfile(base_url="https://example.test"))

    assert str(exc_info.value) == "\n".join(
        [
            "Remote query job-1 failed.",
            "Summary: Remote query submission failed.",
            "Stage: submission",
            "Error type: FacilityUnauthorizedError",
            "Detail:",
            "  Keycloak ID token not valid or expired. Try to re-authenticate.",
        ]
    )


def test_remote_query_fetch_hydrates_immediate_succeeded_submit_status(
    monkeypatch, tmp_path
):
    calls = []

    def urlopen(req, timeout=None, context=None):
        calls.append(req.full_url)
        if req.full_url.endswith("/queries") and req.get_method() == "POST":
            return _Response(
                json.dumps({"job_id": "job-1", "status": "succeeded"}).encode()
            )
        if req.full_url.endswith("/queries/job-1"):
            return _Response(
                RemoteQueryStatus(
                    job_id="job-1",
                    status="succeeded",
                    result_url="https://example.test/results/job-1.hdf5",
                )
                .model_dump_json()
                .encode()
            )
        return _Response(b"fake-hdf5")

    def fake_open(path):
        return Path(path).read_bytes()

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)
    monkeypatch.setattr(oc, "open", fake_open)

    result = oc.remote.open(
        "Frontier-E",
        ["halo_properties"],
        product="snapshot",
        steps=205,
    ).fetch(
        profile=oc.remote.RemoteProfile(
            base_url="https://example.test",
            result_cache_dir=tmp_path,
        )
    )

    assert result == b"fake-hdf5"
    assert calls == [
        "https://example.test/queries",
        "https://example.test/queries/job-1",
        "https://example.test/results/job-1.hdf5",
    ]


def test_remote_query_status_accepts_optional_failure_metadata():
    status = RemoteQueryStatus.model_validate(
        {
            "job_id": "job-1",
            "status": "failed",
            "failure_stage": "execution",
            "error_type": "RuntimeError",
            "error_detail": "boom",
            "stderr_excerpt": "stderr",
            "stdout_excerpt": "stdout",
        }
    )

    assert status.failure_stage == "execution"
    assert status.error_type == "RuntimeError"
    assert status.error_detail == "boom"
    assert status.stderr_excerpt == "stderr"
    assert status.stdout_excerpt == "stdout"


def test_status_display_uses_single_notebook_handle(monkeypatch):
    events = []

    class FakeHandle:
        def display(self, text):
            events.append(("display", text))

        def update(self, text):
            events.append(("update", text))

    class FakeShell:
        pass

    FakeShell.__module__ = "ipykernel.zmqshell"

    monkeypatch.setattr(remote_status_display, "_get_ipython_shell", lambda: FakeShell())
    monkeypatch.setattr(
        remote_status_display,
        "_get_notebook_handle_class",
        lambda: FakeHandle,
    )
    monkeypatch.setattr(
        remote_status_display,
        "_get_notebook_text_renderer",
        lambda: (lambda text: text),
    )

    sink = remote_status_display.create_status_sink()
    sink.emit("first")
    sink.emit("second")
    sink.close()

    assert events == [("display", "first"), ("update", "first\nsecond")]


def test_status_display_falls_back_to_stderr_when_not_in_notebook():
    stream = io.StringIO()

    sink = remote_status_display.create_status_sink(stream=stream)
    sink.emit("fallback")
    sink.close()

    assert isinstance(sink, remote_status_display.StderrStatusSink)
    assert stream.getvalue() == "fallback\n"


def test_remote_client_submit_serializes_structured_and_file_collection_sources(
    monkeypatch,
):
    payloads = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"job_id": "job-1", "status": "queued"}).encode()

    def urlopen(req, timeout=None, context=None):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return Response()

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)

    client = RemoteClient(oc.remote.RemoteProfile(base_url="https://example.test"))
    client.submit(
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        ).into_request()
    )
    client.submit(
        oc.remote.open_collection(
            "LastJourney-Diffsky-COSMOS-2026-02-17",
            open_kwargs={"synth_cores": True},
        )
        .take(10, at="start")
        .into_request()
    )
    client.submit(
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        )
        .with_execution(priority="debug", walltime="00:15:00")
        .into_request()
    )

    assert payloads[0]["source"]["kind"] == "structured_catalog"
    assert payloads[0]["source"]["product"] == "snapshot"
    assert payloads[0]["source"]["steps"] == [205]
    assert payloads[0]["source"]["catalogs"] == ["halo_properties"]

    assert payloads[1]["source"]["kind"] == "file_collection"
    assert payloads[1]["source"]["remote_dataset"] == (
        "LastJourney-Diffsky-COSMOS-2026-02-17"
    )
    assert payloads[1]["source"]["open_kwargs"] == {"synth_cores": True}
    assert "steps" not in payloads[1]["source"]
    assert "catalogs" not in payloads[1]["source"]

    assert payloads[2]["execution"] == {
        "priority": "debug",
        "walltime": "00:15:00",
    }


def test_remote_query_submit_uses_default_profile_when_unconfigured(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("opencosmo.remote.client._default_profile", None)
    monkeypatch.setattr(
        "opencosmo.remote.client.DEFAULT_AUTH_STORAGE_PATH",
        tmp_path / "remote-auth.json",
    )

    calls = []

    def urlopen(req, timeout=None, context=None):
        calls.append(req.full_url)
        return _Response(json.dumps({"job_id": "job-1", "status": "queued"}).encode())

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)

    response = oc.remote.open(
        "Frontier-E",
        ["halo_properties"],
        product="snapshot",
        steps=205,
    ).submit()

    assert response.job_id == "job-1"
    assert isinstance(response, RemoteQueryResponse)
    assert calls == [f"{DEFAULT_REMOTE_BASE_URL}/queries"]


def test_remote_client_uses_stored_auth_and_refreshes_when_expired(
    monkeypatch, tmp_path
):
    storage_path = tmp_path / "remote-auth.json"
    put_entry(
        storage_path,
        StoredRemoteAuth(
            base_url="https://example.test",
            client_id=DEFAULT_AUTH_CLIENT_ID,
            required_scope="scope://remote",
            access_token="stale-token",
            refresh_token="refresh-token",
            expires_at=int(time.time()) - 10,
            dependent_scopes=("scope://facility",),
            session_required_policies=("policy-1",),
        ),
    )

    class FakeNativeAppAuthClient:
        def __init__(self, client_id):
            assert client_id == DEFAULT_AUTH_CLIENT_ID

        def oauth2_refresh_token(self, refresh_token):
            assert refresh_token == "refresh-token"
            return _token_response(
                access_token="fresh-token",
                refresh_token="new-refresh-token",
                scope="scope://remote",
                expires_at=int(time.time()) + 3600,
            )

    class FakeGlobusSDK:
        NativeAppAuthClient = FakeNativeAppAuthClient

    seen_auth = []

    def urlopen(req, timeout=None, context=None):
        seen_auth.append(req.get_header("Authorization"))
        return _Response(
            RemoteQueryStatus(job_id="job-1", status="queued")
            .model_dump_json()
            .encode()
        )

    monkeypatch.setattr(
        "opencosmo.remote._auth_store._get_globus_sdk", lambda: FakeGlobusSDK
    )
    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)

    client = RemoteClient(
        oc.remote.RemoteProfile(
            base_url="https://example.test",
            auth_storage_path=storage_path,
        )
    )
    status = client.get_status("job-1")

    assert status.status == "queued"
    assert seen_auth == ["Bearer fresh-token"]
    stored = json.loads(storage_path.read_text())
    assert stored["entries"]["https://example.test"]["access_token"] == "fresh-token"
    assert (
        stored["entries"]["https://example.test"]["refresh_token"]
        == "new-refresh-token"
    )
    assert stored["entries"]["https://example.test"]["dependent_scopes"] == [
        "scope://facility"
    ]
    assert stored["entries"]["https://example.test"]["session_required_policies"] == [
        "policy-1"
    ]


def test_remote_client_refresh_failure_clears_stored_auth(monkeypatch, tmp_path):
    storage_path = tmp_path / "remote-auth.json"
    put_entry(
        storage_path,
        StoredRemoteAuth(
            base_url="https://example.test",
            client_id=DEFAULT_AUTH_CLIENT_ID,
            required_scope="scope://remote",
            access_token="stale-token",
            refresh_token="refresh-token",
            expires_at=int(time.time()) - 10,
            dependent_scopes=("scope://facility",),
            session_required_policies=("policy-1",),
        ),
    )

    class FakeNativeAppAuthClient:
        def __init__(self, client_id):
            assert client_id == DEFAULT_AUTH_CLIENT_ID

        def oauth2_refresh_token(self, refresh_token):
            raise RuntimeError("revoked")

    class FakeGlobusSDK:
        NativeAppAuthClient = FakeNativeAppAuthClient

    monkeypatch.setattr(
        "opencosmo.remote._auth_store._get_globus_sdk", lambda: FakeGlobusSDK
    )

    client = RemoteClient(
        oc.remote.RemoteProfile(
            base_url="https://example.test",
            auth_storage_path=storage_path,
        )
    )

    with pytest.raises(RemoteError, match="Stored remote login is no longer valid"):
        client.get_status("job-1")

    assert not storage_path.exists()


def test_remote_client_explicit_authorization_header_beats_stored_auth(
    monkeypatch, tmp_path
):
    storage_path = tmp_path / "remote-auth.json"
    put_entry(
        storage_path,
        StoredRemoteAuth(
            base_url="https://example.test",
            client_id=DEFAULT_AUTH_CLIENT_ID,
            required_scope="scope://remote",
            access_token="stored-token",
            refresh_token="refresh-token",
            expires_at=int(time.time()) + 3600,
            dependent_scopes=("scope://facility",),
            session_required_policies=("policy-1",),
        ),
    )

    seen_auth = []

    def urlopen(req, timeout=None, context=None):
        seen_auth.append(req.get_header("Authorization"))
        return _Response(
            RemoteQueryStatus(job_id="job-1", status="queued")
            .model_dump_json()
            .encode()
        )

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)

    client = RemoteClient(
        oc.remote.RemoteProfile(
            base_url="https://example.test",
            headers={"Authorization": "Bearer header-token"},
            auth_storage_path=storage_path,
        )
    )
    client.get_status("job-1")

    assert seen_auth == ["Bearer header-token"]


def test_remote_client_explicit_token_beats_stored_auth(monkeypatch, tmp_path):
    storage_path = tmp_path / "remote-auth.json"
    put_entry(
        storage_path,
        StoredRemoteAuth(
            base_url="https://example.test",
            client_id=DEFAULT_AUTH_CLIENT_ID,
            required_scope="scope://remote",
            access_token="stored-token",
            refresh_token="refresh-token",
            expires_at=int(time.time()) + 3600,
            dependent_scopes=("scope://facility",),
            session_required_policies=("policy-1",),
        ),
    )

    seen_auth = []

    def urlopen(req, timeout=None, context=None):
        seen_auth.append(req.get_header("Authorization"))
        return _Response(
            RemoteQueryStatus(job_id="job-1", status="queued")
            .model_dump_json()
            .encode()
        )

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)

    client = RemoteClient(
        oc.remote.RemoteProfile(
            base_url="https://example.test",
            token="memory-token",
            auth_storage_path=storage_path,
        )
    )
    client.get_status("job-1")

    assert seen_auth == ["Bearer memory-token"]


def test_remote_client_result_download_does_not_send_authorization(
    monkeypatch, tmp_path
):
    seen_headers = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b"fake-hdf5"

    def urlopen(req, timeout=None, context=None):
        seen_headers.append(dict(req.header_items()))
        return Response()

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)

    client = RemoteClient(
        oc.remote.RemoteProfile(
            base_url="https://example.test",
            token="memory-token",
            headers={"X-Test": "1"},
            result_cache_dir=tmp_path,
        )
    )
    output_path = client.download_result(
        RemoteQueryStatus(
            job_id="job-1",
            status="succeeded",
            result_url="https://example.test/results/job-1.hdf5",
        )
    )

    assert output_path.read_bytes() == b"fake-hdf5"
    assert seen_headers == [{"X-test": "1"}]


def test_remote_client_raises_authorization_required(monkeypatch):
    def urlopen(req, timeout=None, context=None):
        raise HTTPError(
            req.full_url,
            403,
            "Forbidden",
            hdrs=None,
            fp=io.BytesIO(
                json.dumps(
                    {
                        "detail": {
                            "error": "authorization_required",
                            "authorization_parameters": {
                                "required_scopes": ["scope://facility"],
                                "session_required_policies": ["policy-1"],
                                "session_message": "consent required",
                                "prompt": "login",
                            },
                        }
                    }
                ).encode()
            ),
        )

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)
    client = RemoteClient(oc.remote.RemoteProfile(base_url="https://example.test"))

    with pytest.raises(RemoteAuthorizationRequired) as exc:
        client.get_status("job-1")

    assert exc.value.required_scopes == ("scope://facility",)
    assert exc.value.session_required_policies == ("policy-1",)
    assert exc.value.session_message == "consent required"
    assert exc.value.prompt == "login"
    assert "opencosmo remote login" in str(exc.value)


def test_remote_client_keeps_generic_403_errors(monkeypatch):
    def urlopen(req, timeout=None, context=None):
        raise HTTPError(
            req.full_url,
            403,
            "Forbidden",
            hdrs=None,
            fp=io.BytesIO(b'{"detail":"forbidden"}'),
        )

    monkeypatch.setattr("opencosmo.remote.client.request.urlopen", urlopen)
    client = RemoteClient(oc.remote.RemoteProfile(base_url="https://example.test"))

    with pytest.raises(RemoteError, match="Remote API request failed: 403"):
        client.get_status("job-1")


def test_execute_remote_query_replays_through_existing_open_and_write(monkeypatch):
    events = []

    class FakeData:
        def filter(self, *masks):
            events.append(("filter", len(masks)))
            return self

        def take(self, n, at="random"):
            events.append(("take", n, at))
            return self

        def select(self, *args, **kwargs):
            events.append(("select", args, kwargs))
            return self

    def fake_open(*paths, **kwargs):
        events.append(("open", paths, kwargs))
        return FakeData()

    def fake_write(path, dataset):
        events.append(("write", Path(path), isinstance(dataset, FakeData)))

    monkeypatch.setattr(oc, "open", fake_open)
    monkeypatch.setattr(oc, "write", fake_write)

    request_data = (
        oc.remote.open(
            "Frontier-E",
            ["halo_properties"],
            product="snapshot",
            steps=205,
        )
        .filter(oc.col("fof_halo_mass") > 1e13)
        .take(10, at="start")
        .select("fof_halo_mass")
        .into_request()
        .model_dump(mode="python")
    )
    request_data["source"]["open_kwargs"] = {"synth_cores": True}
    request = RemoteQueryRequest.model_validate(request_data)

    output_path = execute_remote_query(
        request,
        lambda source: [
            f"{source.remote_dataset}_{source.product}_"
            f"{source.steps[0]}_{source.catalogs[0]}.hdf5"
        ],
        "result.hdf5",
    )

    assert output_path == Path("result.hdf5")
    assert events[0] == (
        "open",
        ("Frontier-E_snapshot_205_halo_properties.hdf5",),
        {"synth_cores": True},
    )
    assert events[-1] == ("write", Path("result.hdf5"), True)


def test_execute_remote_query_replays_file_collection_sources(
    monkeypatch, diffsky_path
):
    events = []
    diffsky_files = (diffsky_path / "lj_475.hdf5", diffsky_path / "lj_487.hdf5")

    class FakeData:
        def take(self, n, at="random"):
            events.append(("take", n, at))
            return self

        def select(self, *args, **kwargs):
            events.append(("select", args, kwargs))
            return self

    def fake_open(*paths, **kwargs):
        events.append(("open", paths, kwargs))
        return FakeData()

    def fake_write(path, dataset):
        events.append(("write", Path(path), isinstance(dataset, FakeData)))

    monkeypatch.setattr(oc, "open", fake_open)
    monkeypatch.setattr(oc, "write", fake_write)

    request = (
        oc.remote.open_collection(
            "LastJourney-Diffsky-COSMOS-2026-02-17",
            open_kwargs={"synth_cores": True},
        )
        .select("ra", "dec")
        .take(5, at="start")
        .into_request()
    )

    def resolver(source):
        events.append(("resolve", source.kind, source.remote_dataset))
        assert isinstance(source, FileCollectionSource)
        return diffsky_files

    output_path = execute_remote_query(request, resolver, "result.hdf5")

    assert output_path == Path("result.hdf5")
    assert events[0] == (
        "resolve",
        "file_collection",
        "LastJourney-Diffsky-COSMOS-2026-02-17",
    )
    assert events[1] == ("open", diffsky_files, {"synth_cores": True})
    assert events[-1] == ("write", Path("result.hdf5"), True)


def test_remote_cli_status(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr(
        "opencosmo.remote.cli._remote_profile",
        lambda base_url: oc.remote.RemoteProfile(
            base_url=base_url or "https://example.test",
            auth_storage_path=tmp_path / "remote-auth.json",
        ),
    )

    result = runner.invoke(cli, ["remote", "status"])

    assert result.exit_code == 0
    payload = _cli_json_output(result.output)
    assert payload["authenticated"] is False
    assert payload["base_url"] == "https://example.test"


def test_remote_cli_login_with_token(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr(
        "opencosmo.remote.cli._remote_profile",
        lambda base_url: oc.remote.RemoteProfile(
            base_url=base_url or "https://example.test",
            auth_storage_path=tmp_path / "remote-auth.json",
        ),
    )

    result = runner.invoke(cli, ["remote", "login", "--token", "secret"])

    assert result.exit_code == 0
    payload = _cli_json_output(result.output)
    assert payload["authenticated"] is True
    assert payload["token_source"] == "memory"
    assert oc.remote.get_profile().token == "secret"


def test_remote_cli_login_with_auth_code(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr(
        "opencosmo.remote.cli._remote_profile",
        lambda base_url: oc.remote.RemoteProfile(
            base_url=base_url or "https://example.test",
            auth_storage_path=tmp_path / "remote-auth.json",
        ),
    )
    monkeypatch.setattr(
        "opencosmo.remote.auth.request.urlopen",
        lambda req, timeout=None, context=None: _Response(
            json.dumps(
                {
                    "auth_provider": "globus",
                    "required_scope": "scope://remote",
                    "dependent_scopes": [],
                    "session_required_policies": [],
                }
            ).encode()
        ),
    )

    class FakeNativeAppAuthClient:
        def __init__(self, client_id):
            assert client_id == DEFAULT_AUTH_CLIENT_ID

        def oauth2_start_flow(self, **kwargs):
            return None

        def oauth2_get_authorize_url(self, **kwargs):
            return "https://auth.globus.org/authorize"

        def oauth2_exchange_code_for_tokens(self, code):
            assert code == "auth-code"
            return _token_response(
                access_token="stored-token",
                refresh_token="refresh-token",
                scope="scope://remote",
                expires_at=int(time.time()) + 3600,
            )

    class FakeGlobusSDK:
        NativeAppAuthClient = FakeNativeAppAuthClient

    monkeypatch.setattr("opencosmo.remote.auth._get_globus_sdk", lambda: FakeGlobusSDK)

    result = runner.invoke(
        cli,
        ["remote", "login", "--auth-code", "auth-code"],
    )

    assert result.exit_code == 0
    payload = _cli_json_output(result.output)
    assert payload["token_source"] == "stored"
    assert payload["required_scope"] == "scope://remote"


def test_remote_cli_login_defaults_to_no_browser(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr(
        "opencosmo.remote.cli._remote_profile",
        lambda base_url: oc.remote.RemoteProfile(
            base_url=base_url or "https://example.test",
            auth_storage_path=tmp_path / "remote-auth.json",
        ),
    )

    observed = {}

    def fake_login(*, token=None, auth_code=None):
        observed["token"] = token
        observed["auth_code"] = auth_code
        return oc.remote.auth.AuthStatus(
            authenticated=True,
            message="Stored remote login completed.",
            base_url="https://example.test",
            token_source="stored",
        )

    monkeypatch.setattr("opencosmo.remote.auth.login", fake_login)

    result = runner.invoke(cli, ["remote", "login", "--auth-code", "auth-code"])

    assert result.exit_code == 0
    assert observed == {
        "token": None,
        "auth_code": "auth-code",
    }


def test_remote_cli_logout(monkeypatch, tmp_path):
    storage_path = tmp_path / "remote-auth.json"
    put_entry(
        storage_path,
        StoredRemoteAuth(
            base_url="https://example.test",
            client_id=DEFAULT_AUTH_CLIENT_ID,
            required_scope="scope://remote",
            access_token="stored-token",
            refresh_token="refresh-token",
            expires_at=1_800_000_000,
        ),
    )
    runner = CliRunner()
    monkeypatch.setattr(
        "opencosmo.remote.cli._remote_profile",
        lambda base_url: oc.remote.RemoteProfile(
            base_url=base_url or "https://example.test",
            auth_storage_path=storage_path,
        ),
    )

    result = runner.invoke(cli, ["remote", "logout"])

    assert result.exit_code == 0
    payload = _cli_json_output(result.output)
    assert payload["authenticated"] is False
    assert not storage_path.exists()


def test_remote_cli_reauth_with_token(monkeypatch, tmp_path):
    runner = CliRunner()
    events: list[tuple[str, str | None, str | None]] = []
    monkeypatch.setattr(
        "opencosmo.remote.cli._remote_profile",
        lambda base_url: oc.remote.RemoteProfile(
            base_url=base_url or "https://example.test",
            auth_storage_path=tmp_path / "remote-auth.json",
        ),
    )

    def fake_logout():
        events.append(("logout", None, None))
        return oc.remote.auth.AuthStatus(
            authenticated=False,
            message="Logged out.",
            base_url="https://example.test",
            token_source="none",
        )

    def fake_login(*, token=None, auth_code=None):
        events.append(("login", token, auth_code))
        return oc.remote.auth.AuthStatus(
            authenticated=True,
            message="Stored remote login completed.",
            base_url="https://example.test",
            token_source="memory",
        )

    monkeypatch.setattr("opencosmo.remote.auth.logout", fake_logout)
    monkeypatch.setattr("opencosmo.remote.auth.login", fake_login)

    result = runner.invoke(cli, ["remote", "reauth", "--token", "secret"])

    assert result.exit_code == 0
    assert events == [
        ("logout", None, None),
        ("login", "secret", None),
    ]
    payload = _cli_json_output(result.output)
    assert payload["authenticated"] is True
    assert payload["token_source"] == "memory"


def test_remote_cli_reauth_with_auth_code(monkeypatch, tmp_path):
    runner = CliRunner()
    events: list[tuple[str, str | None, str | None]] = []
    monkeypatch.setattr(
        "opencosmo.remote.cli._remote_profile",
        lambda base_url: oc.remote.RemoteProfile(
            base_url=base_url or "https://example.test",
            auth_storage_path=tmp_path / "remote-auth.json",
        ),
    )

    def fake_logout():
        events.append(("logout", None, None))
        return oc.remote.auth.AuthStatus(
            authenticated=False,
            message="Logged out.",
            base_url="https://example.test",
            token_source="none",
        )

    def fake_login(*, token=None, auth_code=None):
        events.append(("login", token, auth_code))
        return oc.remote.auth.AuthStatus(
            authenticated=True,
            message="Stored remote login completed.",
            base_url="https://example.test",
            token_source="stored",
        )

    monkeypatch.setattr("opencosmo.remote.auth.logout", fake_logout)
    monkeypatch.setattr("opencosmo.remote.auth.login", fake_login)

    result = runner.invoke(cli, ["remote", "reauth", "--auth-code", "auth-code"])

    assert result.exit_code == 0
    assert events == [
        ("logout", None, None),
        ("login", None, "auth-code"),
    ]
    payload = _cli_json_output(result.output)
    assert payload["authenticated"] is True
    assert payload["token_source"] == "stored"


def test_remote_cli_reauth_clears_existing_store_before_login(monkeypatch, tmp_path):
    storage_path = tmp_path / "remote-auth.json"
    put_entry(
        storage_path,
        StoredRemoteAuth(
            base_url="https://example.test",
            client_id=DEFAULT_AUTH_CLIENT_ID,
            required_scope="scope://remote",
            access_token="stored-token",
            refresh_token="refresh-token",
            expires_at=1_800_000_000,
        ),
    )
    runner = CliRunner()
    monkeypatch.setattr(
        "opencosmo.remote.cli._remote_profile",
        lambda base_url: oc.remote.RemoteProfile(
            base_url=base_url or "https://example.test",
            auth_storage_path=storage_path,
        ),
    )

    def fake_login(*, token=None, auth_code=None):
        assert token is None
        assert auth_code == "auth-code"
        assert not storage_path.exists()
        return oc.remote.auth.AuthStatus(
            authenticated=True,
            message="Stored remote login completed.",
            base_url="https://example.test",
            token_source="stored",
        )

    monkeypatch.setattr("opencosmo.remote.auth.login", fake_login)

    result = runner.invoke(
        cli,
        ["remote", "reauth", "--auth-code", "auth-code"],
    )

    assert result.exit_code == 0
    payload = _cli_json_output(result.output)
    assert payload["authenticated"] is True


def test_remote_cli_cleanup_removes_result_cache_dir(monkeypatch, tmp_path):
    result_cache_dir = tmp_path / "remote-cache"
    output_path = result_cache_dir / "job-123" / "result.hdf5"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"result")

    storage_path = tmp_path / "remote-auth.json"
    put_entry(
        storage_path,
        StoredRemoteAuth(
            base_url="https://example.test",
            client_id=DEFAULT_AUTH_CLIENT_ID,
            required_scope="scope://remote",
            access_token="stored-token",
            refresh_token="refresh-token",
            expires_at=1_800_000_000,
        ),
    )

    runner = CliRunner()
    monkeypatch.setattr(
        "opencosmo.remote.cli.remote_module.get_profile",
        lambda: oc.remote.RemoteProfile(
            base_url="https://example.test",
            result_cache_dir=result_cache_dir,
            auth_storage_path=storage_path,
        ),
    )

    result = runner.invoke(cli, ["remote", "cleanup"])

    assert result.exit_code == 0
    payload = _cli_json_output(result.output)
    assert payload["cache_dir"] == str(result_cache_dir)
    assert payload["removed"] is True
    assert payload["message"] == "Remote result cache directory removed."
    assert not result_cache_dir.exists()
    assert storage_path.exists()


def test_remote_cli_cleanup_noop_when_cache_missing(monkeypatch, tmp_path):
    result_cache_dir = tmp_path / "remote-cache"
    runner = CliRunner()
    monkeypatch.setattr(
        "opencosmo.remote.cli.remote_module.get_profile",
        lambda: oc.remote.RemoteProfile(
            base_url="https://example.test",
            result_cache_dir=result_cache_dir,
        ),
    )

    result = runner.invoke(cli, ["remote", "cleanup"])

    assert result.exit_code == 0
    payload = _cli_json_output(result.output)
    assert payload["cache_dir"] == str(result_cache_dir)
    assert payload["removed"] is False
    assert payload["message"] == "Remote result cache directory is already clean."


def test_remote_cli_cleanup_handles_non_directory_path(monkeypatch, tmp_path):
    result_cache_path = tmp_path / "remote-cache"
    result_cache_path.write_text("cache file", encoding="utf-8")

    runner = CliRunner()
    monkeypatch.setattr(
        "opencosmo.remote.cli.remote_module.get_profile",
        lambda: oc.remote.RemoteProfile(
            base_url="https://example.test",
            result_cache_dir=result_cache_path,
        ),
    )

    result = runner.invoke(cli, ["remote", "cleanup"])

    assert result.exit_code == 0
    payload = _cli_json_output(result.output)
    assert payload["cache_dir"] == str(result_cache_path)
    assert payload["removed"] is True
    assert payload["message"] == "Remote result cache directory removed."
    assert not result_cache_path.exists()


def test_remote_cli_base_url_override(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr(
        "opencosmo.remote.client._default_profile",
        oc.remote.RemoteProfile(
            base_url="https://configured.test",
            auth_storage_path=tmp_path / "remote-auth.json",
        ),
    )

    result = runner.invoke(
        cli,
        ["remote", "status", "--base-url", "https://override.test"],
    )

    assert result.exit_code == 0
    payload = _cli_json_output(result.output)
    assert payload["base_url"] == "https://override.test"


def test_remote_cli_rejects_token_and_auth_code():
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["remote", "login", "--token", "secret", "--auth-code", "auth-code"],
    )

    assert result.exit_code != 0
    assert "--token and --auth-code are mutually exclusive." in result.output


def test_remote_cli_reauth_rejects_token_and_auth_code():
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["remote", "reauth", "--token", "secret", "--auth-code", "auth-code"],
    )

    assert result.exit_code != 0
    assert "--token and --auth-code are mutually exclusive." in result.output


class _Response:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self._data


class _TokenResponse:
    def __init__(self, by_resource_server):
        self.by_resource_server = by_resource_server


def _token_response(
    *,
    access_token: str,
    refresh_token: str,
    scope: str,
    expires_at: int,
) -> _TokenResponse:
    return _TokenResponse(
        {
            "remote.api": {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "scope": scope,
                "expires_at_seconds": expires_at,
            }
        }
    )


def _cli_json_output(output: str) -> dict[str, object]:
    return json.loads(output[output.index("{") :])
