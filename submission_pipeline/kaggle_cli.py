"""Thin wrappers around the `kaggle` command line tool installed in this environment."""

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from submission_pipeline import settings

# the kaggle CLI that `uv sync` installed next to this interpreter (bin/kaggle, or Scripts\kaggle.exe on Windows)
KAGGLE = shutil.which("kaggle", path=str(Path(sys.executable).parent)) or "kaggle"
STATUS = re.compile(r'has status "([^"]+)"')
USERNAME = re.compile(r"username:\s*(\S+)")
# not just "error": slugs like filament-error-analysis-* appear in the URL of every push
PUSH_ERROR = re.compile(r"push error", re.IGNORECASE)
FINISHED = ("complete",)
FAILED = ("error", "cancel", "fail")


def kaggle(*args: str, capture: bool = False) -> str:
    """Run `kaggle <args>`; echo its output unless `capture`, in which case return it."""
    result = subprocess.run(
        [KAGGLE, *args], text=True, stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT, check=False
    )
    if result.returncode != 0 and not capture:
        raise SystemExit(f"kaggle {' '.join(args)} failed")
    return result.stdout or ""


def current_user() -> str:
    """The Kaggle account the CLI is signed in as. Notebooks are pushed under it, so every teammate
    gets their own private copies without any per-person configuration in the repository."""
    match = USERNAME.search(kaggle("config", "view", capture=True))
    if not match or match.group(1) == "None":
        raise SystemExit("not signed in to Kaggle: run `uv run kaggle auth login` first")
    return match.group(1)


def push(build_dir: Path, accelerator: str | None = None) -> None:
    """Push the notebook. A rejected push (rules not accepted, bad source, ...) still exits 0, so read the output."""
    extra = ["--accelerator", accelerator] if accelerator else []
    output = kaggle("kernels", "push", "-p", str(build_dir), *extra, capture=True)
    print(output, end="", flush=True)
    if PUSH_ERROR.search(output):
        raise SystemExit("the push was rejected; nothing is running on Kaggle")


def parse_status(output: str) -> str | None:
    """The status in `<kernel> has status "KernelWorkerStatus.RUNNING"`, lower-cased; None if absent."""
    match = STATUS.search(output)
    return match.group(1).lower() if match else None


def wait(kernel_id: str, poll_sec: int = 30, max_hiccups: int = 20) -> bool:
    """Block until the latest run of the notebook ends; True if it completed.

    A poll that yields no status line (network blip, expired token being refreshed) is retried: a run
    can last hours, and a traceback mentioning "error" must not be mistaken for a failed notebook.
    """
    hiccups = 0
    while True:
        status = parse_status(kaggle("kernels", "status", kernel_id, capture=True))
        stamp = time.strftime("%H:%M:%S")
        if status is None:
            hiccups += 1
            print(f"   {stamp} could not read the status ({hiccups}/{max_hiccups}), retrying", flush=True)
            if hiccups >= max_hiccups:
                raise SystemExit(f"giving up; the run may still be alive: https://www.kaggle.com/code/{kernel_id}")
        else:
            hiccups = 0
            print(f"   {stamp} {status}", flush=True)
            if any(word in status for word in FINISHED):
                return True
            if any(word in status for word in FAILED):
                return False
        time.sleep(poll_sec)


def fetch_log(kernel_id: str, path: Path) -> None:
    """Save the run's log as plain text; the API returns JSON records ({stream_name, time, data})."""
    raw = kaggle("kernels", "logs", kernel_id, capture=True)
    try:
        text = "".join(record.get("data", "") for record in json.loads(raw))
    except ValueError:
        text = raw
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def fetch_outputs(kernel_id: str, out_dir: Path, pattern: str = settings.OUTPUT_PATTERN) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kernel_id, "-p", str(out_dir), "--file-pattern", pattern, "--force", "--quiet")
    # the CLI always adds the raw log next to the outputs; fetch_log already saved a readable one
    (out_dir / f"{kernel_id.split('/')[-1]}.log").unlink(missing_ok=True)


def submission_limits() -> str:
    return kaggle("competitions", "submission-limits", settings.COMPETITION, capture=True).strip()


def submit(csv: Path, message: str) -> None:
    kaggle("competitions", "submit", settings.COMPETITION, "-f", str(csv), "-m", message)


def recent_submissions(n: int = 5) -> str:
    table = kaggle("competitions", "submissions", settings.COMPETITION, capture=True)
    return "\n".join(table.splitlines()[: n + 2])
