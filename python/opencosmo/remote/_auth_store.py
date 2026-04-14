from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opencosmo.remote.client import RemoteProfile

# DEFAULT_AUTH_CLIENT_ID = "132a3ac6-128f-4661-a20b-411a9a376671"
DEFAULT_AUTH_CLIENT_ID = "8b84fc2d-49e9-49ea-b54d-b3a29a70cf31"  # ALCF client ID
DEFAULT_AUTH_STORAGE_PATH = Path.home() / ".config" / "opencosmo" / "remote-auth.json"
_STORE_VERSION = 1
_REFRESH_SKEW_SECONDS = 60


@dataclass(frozen=True)
class StoredRemoteAuth:
    base_url: str
    client_id: str
    required_scope: str
    access_token: str
    refresh_token: str
    expires_at: int
    dependent_scopes: tuple[str, ...] = ()
    session_required_policies: tuple[str, ...] = ()


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def load_store(path: Path) -> dict[str, StoredRemoteAuth]:
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != _STORE_VERSION:
        raise ValueError("Unsupported remote auth store format.")

    entries = data.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("Remote auth store is missing entries.")

    store: dict[str, StoredRemoteAuth] = {}
    for raw_base_url, raw_entry in entries.items():
        if not isinstance(raw_base_url, str) or not isinstance(raw_entry, dict):
            raise ValueError("Remote auth store contains an invalid entry.")
        base_url = normalize_base_url(raw_base_url)
        store[base_url] = StoredRemoteAuth(
            base_url=base_url,
            client_id=_required_str(raw_entry, "client_id"),
            required_scope=_required_str(raw_entry, "required_scope"),
            dependent_scopes=_optional_str_list(raw_entry, "dependent_scopes"),
            session_required_policies=_optional_str_list(
                raw_entry, "session_required_policies"
            ),
            access_token=_required_str(raw_entry, "access_token"),
            refresh_token=_required_str(raw_entry, "refresh_token"),
            expires_at=_required_int(raw_entry, "expires_at"),
        )
    return store


def save_store(path: Path, data: dict[str, StoredRemoteAuth]) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    os.chmod(parent, 0o700)

    payload = {
        "version": _STORE_VERSION,
        "entries": {
            base_url: {
                key: value for key, value in asdict(entry).items() if key != "base_url"
            }
            for base_url, entry in sorted(data.items())
        },
    }

    fd, tmp_name = tempfile.mkstemp(dir=parent, prefix=".remote-auth-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def get_entry(path: Path, base_url: str) -> StoredRemoteAuth | None:
    return load_store(path).get(normalize_base_url(base_url))


def put_entry(path: Path, entry: StoredRemoteAuth) -> None:
    store = load_store(path)
    store[normalize_base_url(entry.base_url)] = entry
    save_store(path, store)


def delete_entry(path: Path, base_url: str) -> None:
    if not path.exists():
        return
    store = load_store(path)
    store.pop(normalize_base_url(base_url), None)
    if store:
        save_store(path, store)
    else:
        path.unlink()


def entry_from_token_response(
    *,
    base_url: str,
    client_id: str,
    required_scope: str,
    dependent_scopes: tuple[str, ...] = (),
    session_required_policies: tuple[str, ...] = (),
    token_response: Any,
    fallback_refresh_token: str | None = None,
) -> StoredRemoteAuth:
    for token_data in _resource_server_tokens(token_response).values():
        if not isinstance(token_data, dict):
            continue
        scopes = str(token_data.get("scope", "")).split()
        if required_scope not in scopes:
            continue
        access_token = _required_str(token_data, "access_token")
        refresh_token = str(
            token_data.get("refresh_token") or fallback_refresh_token or ""
        )
        if not refresh_token:
            raise ValueError("Globus login did not return a refresh token.")
        expires_at = int(
            token_data.get("expires_at_seconds") or token_data.get("expires_at") or 0
        )
        if expires_at <= 0:
            raise ValueError("Globus login did not return token expiry metadata.")
        return StoredRemoteAuth(
            base_url=normalize_base_url(base_url),
            client_id=client_id,
            required_scope=required_scope,
            dependent_scopes=dependent_scopes,
            session_required_policies=session_required_policies,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
        )
    raise ValueError(
        "Globus login did not return a token for the required remote scope."
    )


def resolve_access_token(profile: RemoteProfile) -> str | None:
    entry = get_entry(profile.auth_storage_path, profile.base_url)
    if entry is None:
        return None
    if entry.expires_at > int(time.time()) + _REFRESH_SKEW_SECONDS:
        return entry.access_token

    try:
        globus_sdk = _get_globus_sdk()
        auth_client = globus_sdk.NativeAppAuthClient(profile.auth_client_id)
        token_response = auth_client.oauth2_refresh_token(entry.refresh_token)
        refreshed = entry_from_token_response(
            base_url=entry.base_url,
            client_id=entry.client_id,
            required_scope=entry.required_scope,
            dependent_scopes=entry.dependent_scopes,
            session_required_policies=entry.session_required_policies,
            token_response=token_response,
            fallback_refresh_token=entry.refresh_token,
        )
    except Exception as exc:
        delete_entry(profile.auth_storage_path, entry.base_url)
        from opencosmo.remote.client import RemoteError

        raise RemoteError(
            "Stored remote login is no longer valid. Run oc.remote.auth.login()."
        ) from exc

    put_entry(profile.auth_storage_path, refreshed)
    return refreshed.access_token


def _resource_server_tokens(token_response: Any) -> dict[str, Any]:
    by_resource_server = getattr(token_response, "by_resource_server", None)
    if by_resource_server is None and isinstance(token_response, dict):
        by_resource_server = token_response.get("by_resource_server")
    if not isinstance(by_resource_server, dict):
        raise ValueError("Invalid Globus token response.")
    return by_resource_server


def _get_globus_sdk():
    try:
        import globus_sdk
    except ImportError as exc:
        raise RuntimeError(
            "globus-sdk is required for interactive remote login."
        ) from exc
    return globus_sdk


def _required_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Remote auth store entry is missing {key}.")
    return value


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Remote auth store entry is missing {key}.")
    return value


def _optional_str_list(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"Remote auth store entry has invalid {key}.")
    return tuple(value)
