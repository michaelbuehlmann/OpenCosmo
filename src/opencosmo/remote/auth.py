from __future__ import annotations

import json
import webbrowser
from dataclasses import replace
from typing import TYPE_CHECKING, Literal
from urllib import request
from urllib.error import HTTPError, URLError

from pydantic import BaseModel, ConfigDict

from opencosmo.remote._auth_store import (
    delete_entry,
    entry_from_token_response,
    get_entry,
    normalize_base_url,
    put_entry,
)
from opencosmo.remote.client import RemoteError, configure, get_profile

if TYPE_CHECKING:
    from opencosmo.remote._auth_store import StoredRemoteAuth
    from opencosmo.remote.client import RemoteProfile

_AUTH_CODE_REDIRECT_URI = "https://auth.globus.org/v2/web/auth-code"
_AUTH_METADATA_PATH = "/.well-known/opencosmo-remote-auth"


class AuthStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    authenticated: bool
    message: str
    base_url: str | None = None
    token_source: Literal["none", "memory", "stored"] = "none"
    required_scope: str | None = None
    dependent_scopes: tuple[str, ...] = ()
    session_required_policies: tuple[str, ...] = ()
    expires_at: int | None = None


class _RemoteAuthMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    auth_provider: Literal["globus"]
    required_scope: str
    dependent_scopes: tuple[str, ...] = ()
    session_required_policies: tuple[str, ...] = ()


def login(
    token: str | None = None,
    *,
    auth_code: str | None = None,
    open_browser: bool = True,
) -> AuthStatus:
    """
    Configure a bearer token for the active remote profile.
    """
    profile = get_profile()
    base_url = normalize_base_url(profile.base_url)

    if token is None:
        metadata = _auth_metadata(profile)
        requested_scopes = _requested_scopes(metadata)

        globus_sdk = _get_globus_sdk()
        auth_client = globus_sdk.NativeAppAuthClient(profile.auth_client_id)
        auth_client.oauth2_start_flow(
            requested_scopes=requested_scopes,
            redirect_uri=_AUTH_CODE_REDIRECT_URI,
            refresh_tokens=True,
        )
        authorize_url_params = {}
        if metadata.session_required_policies:
            authorize_url_params = {
                "session_required_policies": metadata.session_required_policies,
                "prompt": "login",
            }
        authorize_url = auth_client.oauth2_get_authorize_url(**authorize_url_params)
        print(
            "Open this URL in your browser to authorize OpenCosmo Remote:\n"
            f"{authorize_url}"
        )
        if open_browser:
            try:
                webbrowser.open(authorize_url)
            except Exception:
                pass
        if auth_code is None:
            auth_code = input("Enter the Globus authorization code: ").strip()
        if not auth_code:
            raise RemoteError("No authorization code provided.")

        try:
            token_response = auth_client.oauth2_exchange_code_for_tokens(auth_code)
            entry = entry_from_token_response(
                base_url=base_url,
                client_id=profile.auth_client_id,
                required_scope=metadata.required_scope,
                dependent_scopes=metadata.dependent_scopes,
                session_required_policies=metadata.session_required_policies,
                token_response=token_response,
            )
        except ValueError as exc:
            raise RemoteError(str(exc)) from exc
        except Exception as exc:
            raise RemoteError(
                "Interactive remote login failed during Globus token exchange."
            ) from exc

        put_entry(profile.auth_storage_path, entry)
        configure(replace(profile, token=None))
        return AuthStatus(
            authenticated=True,
            message="Stored remote login completed.",
            base_url=base_url,
            token_source="stored",
            required_scope=entry.required_scope,
            dependent_scopes=entry.dependent_scopes,
            session_required_policies=entry.session_required_policies,
            expires_at=entry.expires_at,
        )

    configure(replace(profile, token=token))
    return AuthStatus(
        authenticated=True,
        message="A remote auth token is set.",
        base_url=base_url,
        token_source="memory",
    )


def status() -> AuthStatus:
    """
    Report whether the active remote profile currently has an auth token.
    """
    try:
        profile = get_profile()
    except RemoteError:
        return AuthStatus(
            authenticated=False,
            message="No remote profile configured.",
        )

    base_url = normalize_base_url(profile.base_url)

    if profile.token is not None:
        return AuthStatus(
            authenticated=True,
            message="A remote auth token is set.",
            base_url=base_url,
            token_source="memory",
        )

    entry = _stored_entry(profile)
    if entry is not None:
        return AuthStatus(
            authenticated=True,
            message="A stored remote login is available.",
            base_url=base_url,
            token_source="stored",
            required_scope=entry.required_scope,
            dependent_scopes=entry.dependent_scopes,
            session_required_policies=entry.session_required_policies,
            expires_at=entry.expires_at,
        )
    return AuthStatus(
        authenticated=False,
        message="No remote auth token is configured.",
        base_url=base_url,
    )


def logout() -> AuthStatus:
    """
    Clear the bearer token from the active remote profile.
    """
    profile = get_profile()
    delete_entry(profile.auth_storage_path, profile.base_url)
    configure(replace(profile, token=None))
    return AuthStatus(
        authenticated=False,
        message="No remote auth token is configured.",
        base_url=normalize_base_url(profile.base_url),
        token_source="none",
    )


def _auth_metadata(profile: RemoteProfile) -> _RemoteAuthMetadata:
    url = f"{normalize_base_url(profile.base_url)}{_AUTH_METADATA_PATH}"
    req = request.Request(url, headers={"Accept": "application/json"})
    try:
        with request.urlopen(
            req,
            timeout=profile.timeout_s,
            context=_ssl_context(profile.verify_ssl),
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RemoteError(
            f"Failed to discover remote auth metadata: {exc.code} {detail}"
        ) from exc
    except (URLError, json.JSONDecodeError) as exc:
        raise RemoteError(f"Failed to discover remote auth metadata: {exc}") from exc

    if not isinstance(payload, dict):
        raise RemoteError("Failed to discover remote auth metadata: invalid payload.")
    return _parse_auth_metadata(payload)


def _stored_entry(profile: RemoteProfile) -> StoredRemoteAuth | None:
    try:
        return get_entry(profile.auth_storage_path, profile.base_url)
    except ValueError as exc:
        raise RemoteError(f"Failed to load remote auth store: {exc}") from exc


def _get_globus_sdk():
    try:
        import globus_sdk
    except ImportError as exc:
        raise RemoteError(
            "globus-sdk is required for interactive remote login."
        ) from exc
    return globus_sdk


def _ssl_context(verify_ssl: bool):
    if verify_ssl:
        return None
    import ssl

    return ssl._create_unverified_context()


def _parse_auth_metadata(payload: dict[str, object]) -> _RemoteAuthMetadata:
    auth_provider = payload.get("auth_provider")
    if auth_provider != "globus":
        raise RemoteError("Remote service auth metadata has invalid auth_provider.")

    required_scope = payload.get("required_scope")
    if not isinstance(required_scope, str) or not required_scope:
        raise RemoteError("Remote service auth metadata is missing required_scope.")

    dependent_scopes = _required_str_tuple(payload, "dependent_scopes")
    session_required_policies = _required_str_tuple(
        payload, "session_required_policies"
    )
    return _RemoteAuthMetadata(
        auth_provider="globus",
        required_scope=required_scope,
        dependent_scopes=dependent_scopes,
        session_required_policies=session_required_policies,
    )


def _required_str_tuple(payload: dict[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise RemoteError(f"Remote service auth metadata has invalid {key}.")
    return _dedupe_strings(tuple(value))


def _requested_scopes(metadata: _RemoteAuthMetadata) -> tuple[str, ...]:
    return _dedupe_strings((metadata.required_scope, *metadata.dependent_scopes))


def _dedupe_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return tuple(output)
