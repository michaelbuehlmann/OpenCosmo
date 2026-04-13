from __future__ import annotations

import html
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Protocol

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
    def __init__(
        self,
        handle,
        renderer: Callable[[str], Any] | None = None,
    ):
        self.__handle = handle
        self.__renderer = (
            _get_notebook_text_renderer() if renderer is None else renderer
        )
        self.__lines: list[str] = []
        self.__shown = False

    def emit(self, text: str) -> None:
        self.__lines.append(text)
        rendered = self.__renderer("\n".join(self.__lines))
        if self.__shown:
            self.__handle.update(rendered)
            return
        self.__handle.display(rendered)
        self.__shown = True

    def close(self) -> None:
        return None


class RemoteStatusReporter:
    def __init__(self, job_id: str, sink: StatusSink | None = None):
        self.__job_id = job_id
        self.__sink = create_status_sink() if sink is None else sink
        self.__submitted_at: datetime | None = None
        self.__last_observed_at: datetime | None = None
        self.__last_status: str | None = None

    def emit_submitted(self) -> None:
        observed_at = _now()
        self.__submitted_at = observed_at
        self.__last_observed_at = observed_at
        self.__last_status = "submitted"
        self.__sink.emit(
            format_status_event(
                self.__job_id,
                "submitted",
                observed_at=observed_at,
            )
        )

    def emit_status(self, status: RemoteQueryStatus) -> None:
        observed_at = _now()
        self.__sink.emit(
            format_status_event(
                status.job_id,
                status.status,
                message=_status_summary(status),
                observed_at=observed_at,
                elapsed_s=_elapsed_seconds(self.__last_observed_at, observed_at),
                total_elapsed_s=_elapsed_seconds(self.__submitted_at, observed_at),
                previous_status=self.__last_status,
            )
        )
        self.__last_observed_at = observed_at
        self.__last_status = status.status

    def close(self) -> None:
        self.__sink.close()


def create_status_sink(stream=None) -> StatusSink:
    shell = _get_ipython_shell()
    if shell is not None and _is_notebook_shell(shell):
        handle_class = _get_notebook_handle_class()
        if handle_class is not None:
            return NotebookStatusSink(
                handle_class(),
                renderer=_get_notebook_text_renderer(),
            )
    return StderrStatusSink(stream=stream)


def format_status_event(
    job_id: str,
    status: str,
    *,
    message: str | None = None,
    observed_at: datetime | None = None,
    elapsed_s: float | None = None,
    total_elapsed_s: float | None = None,
    previous_status: str | None = None,
) -> str:
    timestamp = (observed_at or _now()).strftime("%H:%M:%S")
    suffix = f": {message}" if message else ""
    duration_suffix = _format_duration_suffix(
        elapsed_s=elapsed_s,
        total_elapsed_s=total_elapsed_s,
        previous_status=previous_status,
    )
    return f"[{timestamp}] Remote query {job_id} {status}{suffix}{duration_suffix}"


def _status_summary(status: RemoteQueryStatus) -> str | None:
    if status.message:
        return status.message
    if status.error_type and status.error_detail:
        return f"{status.error_type}: {status.error_detail}"
    if status.status == "failed":
        return "Remote query failed."
    return None


def _now() -> datetime:
    return datetime.now().astimezone()


def _elapsed_seconds(
    started_at: datetime | None,
    ended_at: datetime,
) -> float | None:
    if started_at is None:
        return None
    return max((ended_at - started_at).total_seconds(), 0.0)


def _format_duration_suffix(
    *,
    elapsed_s: float | None,
    total_elapsed_s: float | None,
    previous_status: str | None,
) -> str:
    parts = []
    if elapsed_s is not None and previous_status is not None:
        parts.append(f"{_format_duration(elapsed_s)} since {previous_status}")
    elif elapsed_s is not None:
        parts.append(f"{_format_duration(elapsed_s)} elapsed")
    if total_elapsed_s is not None:
        parts.append(f"{_format_duration(total_elapsed_s)} total")
    if not parts:
        return ""
    return f" ({', '.join(parts)})"


def _format_duration(seconds: float) -> str:
    rounded = round(max(seconds, 0.0), 1)
    if rounded < 60:
        return _format_seconds_value(rounded)

    whole_seconds = int(rounded)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes}m")
    second_value = rounded - (hours * 3600 + minutes * 60)
    if second_value or not parts:
        parts.append(_format_seconds_value(second_value))
    return " ".join(parts)


def _format_seconds_value(seconds: float) -> str:
    text = f"{seconds:.1f}".rstrip("0").rstrip(".")
    return f"{text}s"


def _get_notebook_text_renderer() -> Callable[[str], Any]:
    try:
        from IPython.display import HTML
    except ImportError:
        return lambda text: text

    def render(text: str):
        # Render notebook status as plain preformatted text so frontends do not
        # show Python string repr quotes around each update.
        return HTML(f"<pre>{html.escape(text)}</pre>")

    return render


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
