"""Non-blocking JSONL trial recording with a ROS-friendly interface."""

from __future__ import annotations

import hashlib
import json
import queue
import re
import shutil
import threading
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol


RECORDING_SCHEMA = "drobot.rl-trial.v1"
_RECORDING_ID = re.compile(r"^[0-9]{8}T[0-9]{6}(?:[0-9]{6})?Z-[0-9a-f]{8}$")


def sha256_file(path: Path) -> str | None:
    """Return a file hash without making missing optional metadata fatal."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class TrialRecorder(Protocol):
    """Passive recorder boundary shared by JSONL and future rosbag2 backends."""

    def start_trial(self, metadata: Mapping[str, Any]) -> str: ...

    def record_sample(self, payload: Mapping[str, Any]) -> None: ...

    def record_event(self, payload: Mapping[str, Any]) -> None: ...

    def finish_trial(
        self,
        *,
        status: str,
        error: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None: ...

    def status(self) -> dict[str, Any]: ...


@dataclass
class _ActiveTrial:
    recording_id: str
    directory: Path
    metadata: dict[str, Any]
    messages: queue.Queue[tuple[str, dict[str, Any]]]
    stop_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    finish: dict[str, Any] = field(default_factory=dict)
    dropped_samples: int = 0
    dropped_events: int = 0
    sample_count: int = 0
    event_count: int = 0
    writer_error: str | None = None
    thread: threading.Thread | None = None


class JsonlTrialRecorder:
    """Record one active trial while filesystem work runs on a writer thread.

    ``record_sample`` and ``record_event`` only perform a bounded, non-blocking
    queue insertion. A future ROS 2/rosbag2 recorder can provide the same small
    method surface without changing the policy loop.
    """

    def __init__(self, root: Path, *, queue_size: int = 4096) -> None:
        if queue_size < 128:
            raise ValueError("recording queue_size must be at least 128")
        self.root = root.expanduser().resolve()
        self.queue_size = queue_size
        self._lock = threading.RLock()
        self._active: _ActiveTrial | None = None
        self._known: dict[str, _ActiveTrial] = {}
        self._last_error: str | None = None

    def start_trial(self, metadata: Mapping[str, Any]) -> str:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("An RL recording is already active")
            now = datetime.now(UTC)
            recording_id = (
                now.strftime("%Y%m%dT%H%M%S%fZ") + f"-{uuid.uuid4().hex[:8]}"
            )
            directory = self.root / recording_id
            directory.mkdir(parents=True, exist_ok=False)
            document = {
                "schema": RECORDING_SCHEMA,
                "recording_id": recording_id,
                "label": "",
                "created_at": now.isoformat().replace("+00:00", "Z"),
                "status": "recording",
                "files": {
                    "samples": "samples.jsonl",
                    "events": "events.jsonl",
                    "metadata": "metadata.json",
                },
                **dict(metadata),
            }
            trial = _ActiveTrial(
                recording_id=recording_id,
                directory=directory,
                metadata=document,
                messages=queue.Queue(maxsize=self.queue_size),
            )
            thread = threading.Thread(
                target=self._writer,
                args=(trial,),
                name=f"drobot-recorder-{recording_id[-8:]}",
                daemon=True,
            )
            trial.thread = thread
            self._active = trial
            self._known[recording_id] = trial
            self._last_error = None
            thread.start()
            return recording_id

    def record_sample(self, payload: Mapping[str, Any]) -> None:
        self._enqueue("sample", payload)

    def record_event(self, payload: Mapping[str, Any]) -> None:
        self._enqueue("event", payload)

    def _enqueue(self, kind: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            trial = self._active
        if trial is None:
            return
        try:
            trial.messages.put_nowait((kind, dict(payload)))
        except queue.Full:
            with trial.lock:
                if kind == "sample":
                    trial.dropped_samples += 1
                else:
                    trial.dropped_events += 1

    def finish_trial(
        self,
        *,
        status: str,
        error: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            trial = self._active
            if trial is None:
                return
            self._active = None
        with trial.lock:
            trial.finish = {
                "status": status,
                "error": error,
                "details": dict(details or {}),
                "ended_at": _utc_now(),
            }
        trial.stop_event.set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            trial = self._active
            last_error = self._last_error
        if trial is None:
            return {
                "active": False,
                "recording_id": None,
                "dropped_samples": 0,
                "error": last_error,
                "storage_directory": str(self.root),
            }
        with trial.lock:
            return {
                "active": True,
                "recording_id": trial.recording_id,
                "queued_items": trial.messages.qsize(),
                "dropped_samples": trial.dropped_samples,
                "dropped_events": trial.dropped_events,
                "error": trial.writer_error,
                "storage_directory": str(self.root),
            }

    def list_recordings(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        recordings: list[dict[str, Any]] = []
        for directory in self.root.iterdir():
            if not directory.is_dir() or not _RECORDING_ID.fullmatch(directory.name):
                continue
            try:
                finalizing = self._is_finalizing(directory.name)
                metadata = json.loads(
                    (directory / "metadata.json").read_text(encoding="utf-8")
                )
                if metadata.get("status") == "recording" and not finalizing:
                    metadata = self._recover_orphan(directory, metadata)
            except (OSError, ValueError):
                continue
            archive = self.root / f"{directory.name}.zip"
            recordings.append(
                {
                    "recording_id": directory.name,
                    "label": str(metadata.get("label", "")),
                    "created_at": metadata.get("created_at"),
                    "ended_at": metadata.get("ended_at"),
                    "status": metadata.get("status", "unknown"),
                    "sample_count": int(metadata.get("sample_count", 0)),
                    "dropped_samples": int(metadata.get("dropped_samples", 0)),
                    "duration_s": metadata.get("trial", {}).get("duration_s"),
                    "forward_m_s": metadata.get("trial", {}).get("forward_m_s"),
                    "archive_ready": archive.is_file() or not finalizing,
                    "archive_bytes": (
                        archive.stat().st_size if archive.is_file() else 0
                    ),
                }
            )
        return sorted(
            recordings,
            key=lambda recording: str(recording.get("created_at") or ""),
            reverse=True,
        )

    def rename(self, recording_id: str, label: str) -> None:
        directory = self._directory(recording_id)
        cleaned = label.strip()
        if len(cleaned) > 80 or any(ord(character) < 32 for character in cleaned):
            raise ValueError("Recording label must be at most 80 printable characters")
        self._require_finalized(recording_id)
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") == "recording":
            metadata = self._recover_orphan(directory, metadata)
        metadata["label"] = cleaned
        _write_json(metadata_path, metadata)
        self._build_archive(directory)

    def delete(self, recording_id: str) -> None:
        directory = self._directory(recording_id)
        self._require_finalized(recording_id)
        shutil.rmtree(directory)
        archive = self.root / f"{recording_id}.zip"
        if archive.exists():
            archive.unlink()
        with self._lock:
            self._known.pop(recording_id, None)

    def archive_path(self, recording_id: str) -> Path:
        directory = self._directory(recording_id)
        self._require_finalized(recording_id)
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") == "recording":
            self._recover_orphan(directory, metadata)
        archive = self.root / f"{recording_id}.zip"
        if not archive.is_file():
            self._build_archive(directory)
        return archive

    def close(self, timeout_s: float = 2.0) -> None:
        self.finish_trial(status="interrupted", error="Dashboard process stopped")
        with self._lock:
            threads = [trial.thread for trial in self._known.values()]
        for thread in threads:
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=timeout_s)

    def _directory(self, recording_id: str) -> Path:
        if not _RECORDING_ID.fullmatch(recording_id):
            raise ValueError("Invalid recording ID")
        directory = self.root / recording_id
        if not directory.is_dir():
            raise FileNotFoundError(f"Recording not found: {recording_id}")
        return directory

    def _require_finalized(self, recording_id: str) -> None:
        if self._is_finalizing(recording_id):
            raise RuntimeError("Recording is still being finalized")

    def _is_finalizing(self, recording_id: str) -> bool:
        with self._lock:
            trial = self._known.get(recording_id)
        return bool(
            trial is not None
            and trial.thread is not None
            and trial.thread.is_alive()
        )

    def _recover_orphan(
        self,
        directory: Path,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Make a power-loss recording explicit and downloadable on restart."""

        def line_count(path: Path) -> int:
            if not path.is_file():
                return 0
            with path.open("rb") as stream:
                return sum(1 for _line in stream)

        metadata.update(
            status="interrupted",
            error="Dashboard stopped before this recording was finalized",
            ended_at=_utc_now(),
            sample_count=line_count(directory / "samples.jsonl"),
            event_count=line_count(directory / "events.jsonl"),
        )
        _write_json(directory / "metadata.json", metadata)
        return metadata

    def _writer(self, trial: _ActiveTrial) -> None:
        samples_path = trial.directory / "samples.jsonl"
        events_path = trial.directory / "events.jsonl"
        try:
            _write_json(trial.directory / "metadata.json", trial.metadata)
            with samples_path.open("w", encoding="utf-8", buffering=1) as samples, (
                events_path.open("w", encoding="utf-8", buffering=1)
            ) as events:
                while not trial.stop_event.is_set() or not trial.messages.empty():
                    try:
                        kind, payload = trial.messages.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    stream = samples if kind == "sample" else events
                    stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
                    with trial.lock:
                        if kind == "sample":
                            trial.sample_count += 1
                        else:
                            trial.event_count += 1
            with trial.lock:
                finish = dict(trial.finish)
                trial.metadata.update(
                    status=finish.get("status", "interrupted"),
                    error=finish.get("error"),
                    ended_at=finish.get("ended_at", _utc_now()),
                    result=finish.get("details", {}),
                    sample_count=trial.sample_count,
                    event_count=trial.event_count,
                    dropped_samples=trial.dropped_samples,
                    dropped_events=trial.dropped_events,
                )
            _write_json(trial.directory / "metadata.json", trial.metadata)
            self._build_archive(trial.directory)
        except Exception as exc:
            with trial.lock:
                trial.writer_error = str(exc)
            with self._lock:
                self._last_error = f"Recording {trial.recording_id}: {exc}"
            try:
                trial.metadata.update(
                    status="recording-error",
                    recording_error=str(exc),
                    ended_at=_utc_now(),
                    sample_count=trial.sample_count,
                    event_count=trial.event_count,
                    dropped_samples=trial.dropped_samples,
                    dropped_events=trial.dropped_events,
                )
                _write_json(trial.directory / "metadata.json", trial.metadata)
            except OSError:
                pass

    def _build_archive(self, directory: Path) -> None:
        archive = self.root / f"{directory.name}.zip"
        temporary = archive.with_suffix(".zip.tmp")
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as bundle:
            for name in ("metadata.json", "samples.jsonl", "events.jsonl"):
                path = directory / name
                if path.is_file():
                    bundle.write(path, arcname=f"{directory.name}/{name}")
        temporary.replace(archive)
