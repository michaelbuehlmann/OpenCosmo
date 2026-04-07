from __future__ import annotations

import ssl
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping
from urllib import request
from urllib.error import HTTPError, URLError

from opencosmo.remote.protocol import (
    RemoteQueryAccepted,
    RemoteQueryStatus,
)

if TYPE_CHECKING:
    from opencosmo.remote.protocol import RemoteQueryRequest


class RemoteError(RuntimeError):
    pass


class RemoteJobFailed(RemoteError):
    pass


@dataclass(frozen=True)
class RemoteProfile:
    base_url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    token: str | None = None
    timeout_s: float = 30.0
    poll_interval_s: float = 2.0
    verify_ssl: bool = True
    result_cache_dir: Path = Path.home() / ".cache" / "opencosmo" / "remote"

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
        raise RemoteError(
            "No remote profile configured. Call oc.remote.configure(...)."
        )
    return _default_profile


class RemoteClient:
    def __init__(self, profile: RemoteProfile):
        self.__profile = profile

    def submit(self, query: RemoteQueryRequest) -> RemoteQueryAccepted:
        data = query.model_dump_json().encode("utf-8")
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

        req = request.Request(status.result_url, headers=self.__profile.request_headers)
        try:
            context = self.__ssl_context()
            with request.urlopen(
                req, timeout=self.__profile.timeout_s, context=context
            ) as response:
                output_path.write_bytes(response.read())
        except (HTTPError, URLError) as exc:
            raise RemoteError(f"Failed to download remote query result: {exc}") from exc
        return output_path

    def wait(self, job_id: str, timeout_s: float | None = None) -> RemoteQueryStatus:
        start = time.monotonic()
        while True:
            status = self.get_status(job_id)
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
            headers=self.__profile.request_headers,
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
            raise RemoteError(
                f"Remote API request failed: {exc.code} {detail}"
            ) from exc
        except URLError as exc:
            raise RemoteError(f"Remote API request failed: {exc}") from exc

    def __ssl_context(self):
        if self.__profile.verify_ssl:
            return None
        return ssl._create_unverified_context()


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

    def wait(self, timeout_s: float | None = None) -> RemoteQueryStatus:
        self.__status = self.__client.wait(self.job_id, timeout_s)
        return self.__status

    def get_results(self):
        import opencosmo as oc

        status = self.__status
        if status is None or status.status != "succeeded":
            status = self.wait()
        if status.status == "failed":
            raise RemoteJobFailed(status.message or "Remote query failed.")
        result_path = self.__client.download_result(status)
        return oc.open(result_path)
