from __future__ import annotations

import json
from pathlib import Path

import astropy.units as u
import pytest

import opencosmo as oc
from opencosmo.remote.execution import execute_remote_query
from opencosmo.remote.protocol import RemoteQueryRequest, RemoteQueryStatus


def test_remote_query_serializes_dataset_operations():
    query = (
        oc.remote.open("FrontierE", ["halo_properties"])
        .filter((oc.col("fof_halo_mass") > 1e13) & (oc.col("sod_halo_cdelta") < 10))
        .take(1000, at="random")
        .select("fof_halo_mass", "sod_halo_cdelta")
    )

    request = RemoteQueryRequest.model_validate(query.serialize())

    assert request.source.remote_dataset == "FrontierE"
    assert request.source.catalogs == ("halo_properties",)
    assert [operation.type for operation in request.operations] == [
        "filter",
        "take",
        "select",
    ]
    assert request.operations[0].predicate.type == "compound"
    assert request.operations[2].columns == ("fof_halo_mass", "sod_halo_cdelta")


def test_remote_query_serializes_structure_collection_select_and_units():
    query = (
        oc.remote.open("FrontierE", ["halo_properties", "halo_particles"])
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

    select = request.operations[1]
    assert select.columns_by_dataset["halo_properties"] == (
        "fof_halo_mass",
        "sod_halo_cdelta",
    )
    assert select.columns_by_dataset["dm_particles"] == ("x", "y", "z")

    units = request.operations[2]
    assert units.convention == "physical"
    assert units.dataset_conversions["halo_properties"].columns["fof_halo_mass"] == "kg"
    assert units.dataset_conversions["dm_particles"].conversions["Mpc"] == "km"


def test_remote_auth_placeholder_token_flow():
    oc.remote.configure(oc.remote.RemoteProfile(base_url="https://example.test"))

    assert not oc.remote.auth.status().authenticated
    assert oc.remote.auth.login(token="secret").authenticated
    assert not oc.remote.auth.logout().authenticated

    with pytest.raises(NotImplementedError):
        oc.remote.auth.login()


def test_remote_client_submit_status_and_result_download(monkeypatch, tmp_path):
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
    response = oc.remote.open("FrontierE", ["halo_properties"]).get(profile=profile)
    assert response.get_status().status == "succeeded"
    assert response.get_results() == b"fake-hdf5"
    assert calls == [
        "https://example.test/queries",
        "https://example.test/queries/job-1",
        "https://example.test/results/job-1.hdf5",
    ]


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

    request = (
        oc.remote.open("FrontierE", ["halo_properties"], synth_cores=True)
        .filter(oc.col("fof_halo_mass") > 1e13)
        .take(10, at="start")
        .select("fof_halo_mass")
        .into_request()
    )

    output_path = execute_remote_query(
        request,
        lambda source: [f"{source.remote_dataset}_{source.catalogs[0]}.hdf5"],
        "result.hdf5",
    )

    assert output_path == Path("result.hdf5")
    assert events[0] == (
        "open",
        ("FrontierE_halo_properties.hdf5",),
        {"synth_cores": True},
    )
    assert events[-1] == ("write", Path("result.hdf5"), True)
