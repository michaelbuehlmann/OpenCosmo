from __future__ import annotations

import json
import ssl
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping
from urllib import request
from urllib.error import HTTPError, URLError

from opencosmo.remote._auth_store import (
    DEFAULT_AUTH_CLIENT_ID,
    DEFAULT_AUTH_STORAGE_PATH,
    resolve_access_token,
)
from opencosmo.remote._status_display import RemoteStatusReporter
from opencosmo.remote.protocol import (
    RemoteQueryAccepted,
    RemoteQueryStatus,
)

if TYPE_CHECKING:
    from opencosmo.remote.protocol import RemoteQueryRequest

DEFAULT_REMOTE_BASE_URL = "https://opencosmoremote-production.up.railway.app"


class RemoteError(RuntimeError):
    pass


class RemoteAuthorizationRequired(RemoteError):
    def __init__(
        self,
        *,
        required_scopes: tuple[str, ...],
        session_required_policies: tuple[str, ...],
        session_message: str | None,
        prompt: str | None,
    ) -> None:
        super().__init__(
            "Remote API authorization requires a new login. "
            "Run `opencosmo remote login`."
        )
        self.required_scopes = required_scopes
        self.session_required_policies = session_required_policies
        self.session_message = session_message
        self.prompt = prompt


class RemoteJobFailed(RemoteError):
    pass


@dataclass(frozen=True)
class RemoteProfile:
    base_url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    token: str | None = None
    timeout_s: float = 30.0
    poll_interval_s: float = 15.0
    verify_ssl: bool = True
    result_cache_dir: Path = Path.home() / ".cache" / "opencosmo" / "remote"
    auth_client_id: str = DEFAULT_AUTH_CLIENT_ID
    auth_storage_path: Path = DEFAULT_AUTH_STORAGE_PATH

    @property
    def request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"} | dict(self.headers)
        if self.token is not None:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


_default_profile: RemoteProfile | None = None


def configure(profile: RemoteProfile):
    global _default_profile
    _default_profile = profile


def get_profile(profile: RemoteProfile | None = None) -> RemoteProfile:
    if profile is not None:
        return profile
    if _default_profile is None:
        return RemoteProfile(
            base_url=DEFAULT_REMOTE_BASE_URL,
            auth_storage_path=DEFAULT_AUTH_STORAGE_PATH,
        )
    return _default_profile


class RemoteClient:
    def __init__(self, profile: RemoteProfile):
        self.__profile = profile

    def submit(self, query: RemoteQueryRequest) -> RemoteQueryAccepted:
        data = query.model_dump_json(exclude_none=True).encode("utf-8")
        response = self.__request("POST", "/queries", data=data)
        return RemoteQueryAccepted.model_validate_json(response)

    def get_status(self, job_id: str) -> RemoteQueryStatus:
        response = self.__request("GET", f"/queries/{job_id}")
        return RemoteQueryStatus.model_validate_json(response)

    def download_result(self, status: RemoteQueryStatus) -> Path:
        if status.result_url is None:
            raise RemoteError("Remote query status does not include a result_url.")

        job_dir = self.__profile.result_cache_dir / status.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        output_path = job_dir / "result.hdf5"
        if output_path.exists():
            return output_path

        req = request.Request(status.result_url, headers=self.__download_headers())
        try:
            context = self.__ssl_context()
            with request.urlopen(
                req, timeout=self.__profile.timeout_s, context=context
            ) as response:
                output_path.write_bytes(response.read())
        except (HTTPError, URLError) as exc:
            raise RemoteError(f"Failed to download remote query result: {exc}") from exc
        return output_path

    def wait(
        self,
        job_id: str,
        timeout_s: float | None = None,
        *,
        initial_status: RemoteQueryStatus | None = None,
        on_status: Callable[[RemoteQueryStatus], None] | None = None,
    ) -> RemoteQueryStatus:
        start = time.monotonic()
        last_status = _status_key(initial_status)
        while True:
            status = self.get_status(job_id)
            if on_status is not None and _status_key(status) != last_status:
                on_status(status)
                last_status = _status_key(status)
            if status.status in ("succeeded", "failed"):
                return status
            if timeout_s is not None and time.monotonic() - start >= timeout_s:
                raise TimeoutError(f"Remote query {job_id} did not finish in time.")
            time.sleep(self.__profile.poll_interval_s)

    def __request(self, method: str, path: str, data: bytes | None = None) -> str:
        url = f"{self.__profile.base_url.rstrip('/')}{path}"
        req = request.Request(
            url,
            data=data,
            headers=self.__api_request_headers(),
            method=method,
        )
        try:
            context = self.__ssl_context()
            with request.urlopen(
                req, timeout=self.__profile.timeout_s, context=context
            ) as response:
                return response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            auth_error = _authorization_required_error(exc.code, detail)
            if auth_error is not None:
                raise auth_error from exc
            raise RemoteError(
                f"Remote API request failed: {exc.code} {detail}"
            ) from exc
        except URLError as exc:
            raise RemoteError(f"Remote API request failed: {exc}") from exc

    def __ssl_context(self):
        if self.__profile.verify_ssl:
            return None
        return ssl._create_unverified_context()

    def __api_request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"} | dict(self.__profile.headers)

        if self.__profile.token is not None:
            _strip_authorization_headers(headers)
            headers["Authorization"] = f"Bearer {self.__profile.token}"
            return headers

        if _has_authorization_header(headers):
            return headers

        stored_token = resolve_access_token(self.__profile)
        if stored_token is not None:
            headers["Authorization"] = f"Bearer {stored_token}"
        return headers

    def __download_headers(self) -> dict[str, str]:
        headers = dict(self.__profile.headers)
        _strip_authorization_headers(headers)
        return headers


def _has_authorization_header(headers: Mapping[str, str]) -> bool:
    return any(key.lower() == "authorization" for key in headers)


def _strip_authorization_headers(headers: dict[str, str]) -> None:
    for key in list(headers):
        if key.lower() == "authorization":
            headers.pop(key)


def _authorization_required_error(
    status_code: int, response_body: str
) -> RemoteAuthorizationRequired | None:
    if status_code != 403:
        return None

    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    detail = payload.get("detail")
    if not isinstance(detail, dict) or detail.get("error") != "authorization_required":
        return None

    params = detail.get("authorization_parameters")
    if not isinstance(params, dict):
        return None

    return RemoteAuthorizationRequired(
        required_scopes=_string_tuple(params.get("required_scopes")),
        session_required_policies=_string_tuple(
            params.get("session_required_policies")
        ),
        session_message=_optional_string(params.get("session_message")),
        prompt=_optional_string(params.get("prompt")),
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _status_key(status: RemoteQueryStatus | None) -> tuple[str, str | None] | None:
    if status is None:
        return None
    return (status.status, status.message)


class RemoteQueryResponse:
    def __init__(
        self,
        job_id: str,
        client: RemoteClient,
        initial_status: RemoteQueryStatus | None = None,
    ):
        self.job_id = job_id
        self.__client = client
        self.__status = initial_status

    def get_status(self) -> RemoteQueryStatus:
        self.__status = self.__client.get_status(self.job_id)
        return self.__status

    def wait(
        self, timeout_s: float | None = None, *, show_status: bool = True
    ) -> RemoteQueryStatus:
        if self.__status is not None and self.__status.status in ("succeeded", "failed"):
            return self.__status

        if not show_status:
            self.__status = self.__client.wait(
                self.job_id,
                timeout_s,
                initial_status=self.__status,
            )
            return self.__status

        reporter = RemoteStatusReporter(self.job_id)
        try:
            reporter.emit_submitted()
            if self.__status is not None:
                reporter.emit_status(self.__status)
            self.__status = self.__client.wait(
                self.job_id,
                timeout_s,
                initial_status=self.__status,
                on_status=reporter.emit_status,
            )
            return self.__status
        finally:
            reporter.close()

    def get_results(self, *, show_status: bool = True):
        import opencosmo as oc

        status = self.__status
        if status is None or status.status != "succeeded":
            status = self.wait(show_status=show_status)
        if status.status == "failed":
            raise RemoteJobFailed(status.message or "Remote query failed.")
        result_path = self.__client.download_result(status)
        return oc.open(result_path)
