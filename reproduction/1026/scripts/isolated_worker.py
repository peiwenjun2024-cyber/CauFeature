"""Worker exit helpers for historical TensorFlow jobs."""
import os
import sys
from pathlib import Path


def required_output(name, *, forbidden=()):
    """Require an explicit output root and reject known historical roots."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set {name} to a new run directory.")
    path = Path(value).expanduser().resolve()
    for blocked in forbidden:
        blocked = Path(blocked).resolve()
        if path.is_relative_to(blocked) or blocked.is_relative_to(path):
            raise SystemExit(f"{name} must not overlap historical directory {blocked}")
    return path


def finish_worker(status):
    """Call only after writers close and computation status is determined."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
