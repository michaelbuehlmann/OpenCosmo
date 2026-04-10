from __future__ import annotations

import sys
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from opencosmo.remote.protocol import RemoteQueryStatus


class StatusSink(Protocol):
    def emit(self, text: str) -> None: ...

    def close(self) -> None: ...


class StderrStatusSink:
    def __init__(self, stream=None):
        self.__stream = sys.stderr if stream is None else stream

    def emit(self, text: str) -> None:
        self.__stream.write(f"{text}\n")
        self.__stream.flush()

    def close(self) -> None:
        return None


class NotebookStatusSink:
    def __init__(self, handle):
        self.__handle = handle
        self.__shown = False

    def emit(self, text: str) -> None:
        if self.__shown:
            self.__handle.update(text)
            return
        self.__handle.display(text)
        self.__shown = True

    def close(self) -> None:
        return None


class RemoteStatusReporter:
    def __init__(self, job_id: str, sink: StatusSink | None = None):
        self.__job_id = job_id
        self.__sink = create_status_sink() if sink is None else sink

    def emit_submitted(self) -> None:
        self.__sink.emit(format_status_event(self.__job_id, "submitted"))

    def emit_status(self, status: RemoteQueryStatus) -> None:
        self.__sink.emit(
            format_status_event(
                status.job_id,
                status.status,
                message=status.message,
            )
        )

    def close(self) -> None:
        self.__sink.close()


def create_status_sink(stream=None) -> StatusSink:
    shell = _get_ipython_shell()
    if shell is not None and _is_notebook_shell(shell):
        handle_class = _get_notebook_handle_class()
        if handle_class is not None:
            return NotebookStatusSink(handle_class())
    return StderrStatusSink(stream=stream)


def format_status_event(
    job_id: str,
    status: str,
    *,
    message: str | None = None,
    observed_at: datetime | None = None,
) -> str:
    timestamp = (observed_at or _now()).strftime("%H:%M:%S")
    suffix = f": {message}" if message else ""
    return f"[{timestamp}] Remote query {job_id} {status}{suffix}"


def _now() -> datetime:
    return datetime.now().astimezone()


def _get_ipython_shell():
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def _get_notebook_handle_class():
    try:
        from IPython.display import DisplayHandle
    except ImportError:
        return None
    return DisplayHandle


def _is_notebook_shell(shell) -> bool:
    module = type(shell).__module__
    name = type(shell).__name__
    return module.startswith("ipykernel.") or name == "ZMQInteractiveShell"
