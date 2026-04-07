from __future__ import annotations

from dataclasses import replace

from pydantic import BaseModel, ConfigDict

from opencosmo.remote.client import RemoteError, configure, get_profile


class AuthStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    authenticated: bool
    message: str


def login(token: str | None = None) -> AuthStatus:
    """
    Configure a bearer token for the active remote profile.

    Interactive login is intentionally not implemented until the remote service
    defines its authentication flow.
    """
    if token is None:
        raise NotImplementedError(
            "Interactive remote login is not implemented. Pass token=... or "
            "configure oc.remote.RemoteProfile(token=...)."
        )
    profile = get_profile()
    configure(replace(profile, token=token))
    return status()


def status() -> AuthStatus:
    """
    Report whether the active remote profile currently has an auth token.
    """
    try:
        profile = get_profile()
    except RemoteError:
        return AuthStatus(authenticated=False, message="No remote profile configured.")

    if profile.token is not None:
        return AuthStatus(authenticated=True, message="A remote auth token is set.")
    return AuthStatus(
        authenticated=False, message="No remote auth token is configured."
    )


def logout() -> AuthStatus:
    """
    Clear the bearer token from the active remote profile.
    """
    profile = get_profile()
    configure(replace(profile, token=None))
    return status()
