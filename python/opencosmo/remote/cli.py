from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import click

from opencosmo import remote as remote_module

if TYPE_CHECKING:
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
    if token is not None and auth_code is not None:
        raise click.ClickException("--token and --auth-code are mutually exclusive.")

    status = _call_remote_auth(
        base_url=base_url,
        callback=lambda: remote_module.auth.login(
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
