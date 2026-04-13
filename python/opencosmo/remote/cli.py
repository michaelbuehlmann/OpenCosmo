from __future__ import annotations

import shutil
from dataclasses import replace
from typing import TYPE_CHECKING

import click
from pydantic import BaseModel, ConfigDict

from opencosmo import remote as remote_module

if TYPE_CHECKING:
    from pathlib import Path

    from opencosmo.remote.client import RemoteProfile


@click.group()
def remote():
    pass


@remote.command(name="login")
@click.option("--token", type=str, required=False)
@click.option("--auth-code", type=str, required=False)
@click.option("--base-url", type=str, required=False)
def remote_login(
    token: str | None,
    auth_code: str | None,
    base_url: str | None,
):
    _validate_remote_login_options(token=token, auth_code=auth_code)

    status = _call_remote_auth(
        base_url=base_url,
        callback=lambda: remote_module.auth.login(
            token=token,
            auth_code=auth_code,
        ),
    )
    click.echo(status.model_dump_json(indent=2))


@remote.command(name="reauth")
@click.option("--token", type=str, required=False)
@click.option("--auth-code", type=str, required=False)
@click.option("--base-url", type=str, required=False)
def remote_reauth(
    token: str | None,
    auth_code: str | None,
    base_url: str | None,
):
    _validate_remote_login_options(token=token, auth_code=auth_code)

    status = _call_remote_auth(
        base_url=base_url,
        callback=lambda: remote_module.auth.reauth(
            token=token,
            auth_code=auth_code,
        ),
    )
    click.echo(status.model_dump_json(indent=2))


@remote.command(name="status")
@click.option("--base-url", type=str, required=False)
def remote_status(base_url: str | None):
    status = _call_remote_auth(
        base_url=base_url,
        callback=remote_module.auth.status,
    )
    click.echo(status.model_dump_json(indent=2))


@remote.command(name="logout")
@click.option("--base-url", type=str, required=False)
def remote_logout(base_url: str | None):
    status = _call_remote_auth(
        base_url=base_url,
        callback=remote_module.auth.logout,
    )
    click.echo(status.model_dump_json(indent=2))


@remote.command(name="cleanup")
def remote_cleanup():
    profile = remote_module.get_profile()
    payload = _cleanup_remote_cache(profile.result_cache_dir)
    click.echo(payload.model_dump_json(indent=2))


def _call_remote_auth(*, base_url: str | None, callback):
    profile = _remote_profile(base_url)
    remote_module.configure(profile)
    try:
        return callback()
    except remote_module.RemoteError as exc:
        raise click.ClickException(str(exc)) from exc


def _remote_profile(base_url: str | None) -> RemoteProfile:
    profile = remote_module.get_profile()
    if base_url is None:
        return profile
    return replace(profile, base_url=base_url)


def _validate_remote_login_options(
    *, token: str | None, auth_code: str | None
) -> None:
    if token is not None and auth_code is not None:
        raise click.ClickException("--token and --auth-code are mutually exclusive.")


class _RemoteCleanupStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    cache_dir: str
    removed: bool
    message: str


def _cleanup_remote_cache(path: Path) -> _RemoteCleanupStatus:
    if not path.exists():
        return _RemoteCleanupStatus(
            cache_dir=str(path),
            removed=False,
            message="Remote result cache directory is already clean.",
        )

    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:
        raise click.ClickException(
            f"Failed to remove remote result cache directory {path}: {exc}"
        ) from exc

    return _RemoteCleanupStatus(
        cache_dir=str(path),
        removed=True,
        message="Remote result cache directory removed.",
    )
