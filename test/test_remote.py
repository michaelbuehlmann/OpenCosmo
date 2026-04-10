from __future__ import annotations

import io
import json
import time
from pathlib import Path
from urllib.error import HTTPError

import astropy.units as u
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

import opencosmo as oc
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
    RemoteQueryResponse,
)
from opencosmo.remote.execution import execute_remote_query
from opencosmo.remote.protocol import (
    FileCollectionSource,
    RemoteQueryRequest,
    RemoteQueryStatus,
    StructuredCatalogSource,
)


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
    assert request.operations[2].columns == ("fof_halo_mass", "sod_halo_cdelta")


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


def test_remote_query_request_accepts_legacy_structured_source_without_kind():
    request = RemoteQueryRequest.model_validate(
        {
            "protocol_version": "1.0",
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

    assert isinstance(request.source, StructuredCatalogSource)
    assert request.source.kind == "structured_catalog"
    assert request.source.product == "snapshot"
    assert request.source.steps == (205,)
    assert request.source.catalogs == ("halo_properties",)
    assert request.source.open_kwargs == {"synth_cores": True}


def test_remote_query_request_accepts_explicit_structured_source_kind():
    request = RemoteQueryRequest.model_validate(
        {
            "protocol_version": "1.0",
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


def test_remote_query_request_accepts_file_collection_source_kind():
    request = RemoteQueryRequest.model_validate(
        {
            "protocol_version": "1.0",
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

    browser_urls = []
    monkeypatch.setattr("opencosmo.remote.auth.request.urlopen", urlopen)
    monkeypatch.setattr("opencosmo.remote.auth._get_globus_sdk", lambda: FakeGlobusSDK)
    monkeypatch.setattr(
        "opencosmo.remote.auth.webbrowser.open",
        lambda url: browser_urls.append(url) or False,
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
    assert browser_urls == ["https://auth.globus.org/authorize?scope=scope://remote"]
    assert flow_started == [
        {
            "requested_scopes": (
                "scope://remote",
                "scope://facility",
                "scope://extra",
            ),
            "redirect_uri": "https://auth.globus.org/v2/web/auth-code",
            "refresh_tokens": True,
        }
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

    oc.remote.auth.login(open_browser=False, auth_code="auth-code")

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
        oc.remote.auth.login(open_browser=False, auth_code="unused")


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
        oc.remote.auth.login(open_browser=False, auth_code="unused")


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
        ["remote", "login", "--no-browser", "--auth-code", "auth-code"],
    )

    assert result.exit_code == 0
    payload = _cli_json_output(result.output)
    assert payload["token_source"] == "stored"
    assert payload["required_scope"] == "scope://remote"


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
