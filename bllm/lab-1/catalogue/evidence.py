import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

from pydantic import JsonValue, RootModel

from catalogue.errors import EvidenceError


def json_value(value: object) -> JsonValue:
    return RootModel[JsonValue].model_validate(value).root


def redact(value: JsonValue, secrets: tuple[str, ...]) -> JsonValue:
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {str(redact(key, secrets)): redact(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    return value


@dataclass
class Journal:
    secrets: tuple[str, ...] = field(default=(), repr=False)
    sink: Callable[[dict[str, JsonValue]], None] | None = field(default=None, repr=False)
    events: list[dict[str, JsonValue]] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, request_id: str, event: str, **details: JsonValue) -> None:
        entry: dict[str, JsonValue] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "event": event,
            **details,
        }
        sanitized = {key: redact(value, self.secrets) for key, value in entry.items()}
        with self._lock:
            self.events.append(sanitized)
            if self.sink is not None:
                self.sink(sanitized)


class EvidenceStore:
    def __init__(self, directory: Path, secrets: tuple[str, ...] = ()) -> None:
        self.directory = directory
        self.secrets = secrets
        self._lock = Lock()

    def _destination(self, name: str) -> Path:
        if not name or Path(name).name != name:
            raise EvidenceError("Evidence names must be filenames without directory components.")
        return self.directory / name

    def write_text(self, name: str, text: str) -> None:
        destination = self._destination(name)
        temporary: Path | None = None
        sanitized = str(redact(text, self.secrets))
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(sanitized)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
        except OSError as error:
            raise EvidenceError(f"Could not write evidence file {destination}.") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def remove(self, name: str) -> None:
        path = self._destination(name)
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            raise EvidenceError(f"Could not remove obsolete evidence file {path}.") from error

    def write_json(self, name: str, value: JsonValue) -> None:
        self.write_text(
            name, json.dumps(redact(value, self.secrets), ensure_ascii=False, indent=2) + "\n"
        )

    def read_json(self, name: str) -> JsonValue:
        path = self._destination(name)
        try:
            return json_value(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as error:
            raise EvidenceError(f"Could not read valid JSON evidence from {path}.") from error

    def append_event(self, name: str, event: dict[str, JsonValue]) -> None:
        path = self._destination(name)
        try:
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(redact(event, self.secrets), ensure_ascii=False) + "\n")
        except OSError as error:
            raise EvidenceError(f"Could not append the execution journal at {path}.") from error
