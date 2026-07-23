"""Privacy-critical lifecycle helpers for uploaded service media."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Callable, TypeVar


class UploadPathError(ValueError):
    """Raised when a purported upload is outside Gradio's managed cache."""


class MediaCleanupError(RuntimeError):
    """Raised when terminal media cleanup cannot be completed."""


Result = TypeVar("Result")


def gradio_upload_root(environ: dict[str, str] | None = None) -> Path:
    """Return the resolved cache root used by Gradio for uploaded files."""
    environment = os.environ if environ is None else environ
    configured = environment.get("GRADIO_TEMP_DIR")
    root = Path(configured) if configured else Path(tempfile.gettempdir()) / "gradio"
    return root.expanduser().resolve(strict=False)


def validate_gradio_upload_path(
    value: str | os.PathLike[str],
    *,
    upload_root: Path | None = None,
) -> Path:
    """Resolve one regular nonsymlink upload contained by Gradio's cache."""
    candidate = Path(value)
    if not candidate.is_absolute():
        raise UploadPathError("upload path must be absolute")

    try:
        candidate_stat = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise UploadPathError("upload path is unavailable") from exc

    if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISREG(candidate_stat.st_mode):
        raise UploadPathError("upload path must be a regular nonsymlink file")

    root = (upload_root or gradio_upload_root()).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise UploadPathError("upload path is outside the Gradio cache") from exc
    return resolved


def cleanup_media_files(
    staged_path: str | os.PathLike[str],
    cached_upload_path: str | os.PathLike[str],
    *,
    upload_root: Path | None = None,
) -> None:
    """Remove the private stage and the exact validated Gradio upload."""
    cleanup_failed = False
    staged = Path(staged_path)
    try:
        staged.unlink(missing_ok=True)
    except OSError:
        cleanup_failed = True

    cached: Path | None = None
    try:
        cached_candidate = Path(cached_upload_path)
        if cached_candidate.exists() or cached_candidate.is_symlink():
            cached = validate_gradio_upload_path(
                cached_candidate,
                upload_root=upload_root,
            )
            cached.unlink()
    except (OSError, UploadPathError):
        cleanup_failed = True

    if cached is not None:
        root = (upload_root or gradio_upload_root()).resolve(strict=False)
        parent = cached.parent
        if parent != root:
            try:
                parent.rmdir()
            except OSError:
                # A shared/nonempty cache directory is safe to retain.
                pass

    if cleanup_failed:
        raise MediaCleanupError("media cleanup failed")


def run_with_media_cleanup(
    operation: Callable[[], Result],
    *,
    staged_path: str | os.PathLike[str],
    cached_upload_path: str | os.PathLike[str],
    upload_root: Path | None = None,
) -> Result:
    """Run work while guaranteeing terminal cleanup on every exit path."""
    try:
        return operation()
    finally:
        cleanup_media_files(
            staged_path,
            cached_upload_path,
            upload_root=upload_root,
        )
