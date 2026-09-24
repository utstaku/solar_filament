"""Turn an entry script under solution/ plus the modules it imports into one self-contained Kaggle notebook.

solution/ holds one directory per approach, e.g. solution/unet/, whose scripts are entries:
solution/unet/train.py becomes the notebook `filament-unet-train`, solution/unet/predict.py
`filament-unet-predict`; a directory's main.py is named after the directory alone.
Model architectures live in models/ at the repository root and are shipped the same way.

The generated notebook has three parts:
  1. the entry's own title, then a note saying the notebook is generated;
  2. one `%%writefile` cell per module under solution/ or models/ that the entry imports, directly or
     through another module. Nothing else is shipped, so a notebook never carries code it does not use.
     The cells become real files next to the notebook, so `from common.x import y` works unchanged
     and tracebacks carry the local line numbers;
  3. the cells of the entry script, written in the percent format: a line starting with `# %%`
     opens a code cell, `# %% [markdown]` opens a markdown cell whose lines are comments.

Kaggle's kernel-metadata.json is derived, not hand-written: the notebook's slug and title come from
the entry's path, and run options from an optional magic comment at the top of the entry:

    # kaggle: gpu=true accelerator=NvidiaTeslaT4 kernel_sources=<kaggle user>/filament-unet-train
"""

import ast
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from submission_pipeline import kaggle_cli, settings

CELL_MARKER = "# %%"
OPTIONS_LINE = re.compile(r"^# kaggle:(.*)$", re.MULTILINE)
LIST_OPTIONS = ("kernel_sources", "dataset_sources", "model_sources")


@dataclass
class KernelOptions:
    gpu: bool = False
    internet: bool = True  # pretrained encoder weights are downloaded at run time
    accelerator: str | None = None  # passed to `kaggle kernels push --accelerator`
    kernel_sources: list[str] = field(default_factory=list)
    dataset_sources: list[str] = field(default_factory=list)
    model_sources: list[str] = field(default_factory=list)


@dataclass
class Build:
    name: str  # e.g. `unet-train`; names the notebook, build/<name>/ and outputs/<name>/
    directory: Path  # holds the notebook and kernel-metadata.json, ready for `kaggle kernels push -p`
    kernel_id: str
    options: KernelOptions


def entry_name(entry: Path) -> str:
    """`solution/unet/train.py` -> `unet-train`; `solution/unet/main.py` -> `unet`."""
    try:
        parts = list(entry.resolve().relative_to(settings.SOLUTION_DIR).with_suffix("").parts)
    except ValueError:
        raise SystemExit(f"{entry} is not under {settings.SOLUTION_DIR}") from None
    if len(parts) > 1 and parts[-1] == "main":
        parts.pop()
    return "-".join(parts).replace("_", "-")


def parse_options(source: str) -> KernelOptions:
    options = KernelOptions()
    match = OPTIONS_LINE.search(source)
    for pair in match.group(1).split() if match else []:
        key, _, value = pair.partition("=")
        if key in ("gpu", "internet"):
            setattr(options, key, value.lower() in ("1", "true", "yes", "on"))
        elif key == "accelerator":
            options.accelerator = value
        elif key in LIST_OPTIONS:
            setattr(options, key, [v for v in value.split(",") if v])
        else:
            raise SystemExit(f"unknown option {key!r} in the '# kaggle:' line")
    return options


def kernel_metadata(entry: Path, options: KernelOptions, user: str) -> dict:
    slug = f"{settings.KERNEL_PREFIX}-{entry_name(entry)}"
    return {
        "id": f"{user}/{slug}",
        "title": slug.replace("-", " ").title(),  # Kaggle requires the title to slugify back to the id
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": options.gpu or options.accelerator is not None,
        "enable_tpu": False,
        "enable_internet": options.internet,
        "dataset_sources": options.dataset_sources,
        "competition_sources": [settings.COMPETITION],
        "kernel_sources": options.kernel_sources,
        "model_sources": options.model_sources,
    }


def make_cell(kind: str, text: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def uncomment(line: str) -> str:
    """One line of a markdown cell: drop the leading `# `."""
    return line.removeprefix("# ") if line.startswith("# ") else line.removeprefix("#")


def percent_cells(source: str) -> list[dict]:
    """Split percent-format source into notebook cells."""
    cells, kind, lines = [], "code", []

    def flush():
        text = "\n".join(uncomment(line) for line in lines) if kind == "markdown" else "\n".join(lines)
        if text.strip():
            cells.append(make_cell(kind, text))

    for line in OPTIONS_LINE.sub("", source).splitlines():
        if line.startswith(CELL_MARKER):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        else:
            lines.append(line)
    flush()
    return cells


def resolve(name: str) -> list[Path]:
    """Files behind a dotted name: `common.rle` -> common/__init__.py, common/rle.py under solution/,
    `models.unet` -> models/__init__.py, models/unet.py at the repository root.

    A directory without __init__.py is a package too (Python's namespace packages), so approach
    directories only need one when they have something to put in it.
    """
    parts = name.split(".")
    files, directory = [], settings.SOLUTION_DIR
    if parts[0] in settings.PACKAGES:
        directory = settings.PACKAGES[parts[0]]
        if (directory / "__init__.py").is_file():
            files.append(directory / "__init__.py")
        parts = parts[1:]
    for part in parts:
        if (directory / part).is_dir():
            directory = directory / part
            if (directory / "__init__.py").is_file():
                files.append(directory / "__init__.py")
        else:
            if (directory / f"{part}.py").is_file():
                files.append(directory / f"{part}.py")
            break
    return files


def imported_files(source: str, exclude: tuple[Path, ...] = ()) -> list[Path]:
    """Modules under solution/ that `source` needs, following imports transitively.

    Only absolute imports are understood (`from common.rle import encode`, `from common import rle`).
    `exclude` is for the entry itself, which becomes cells rather than a file.
    """
    found, queue = dict.fromkeys(exclude), [source]
    while queue:
        code = "\n".join(line for line in queue.pop().splitlines() if not line.lstrip().startswith(("%", "!")))
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
            else:
                continue
            for name in names:
                for path in resolve(name):
                    if path not in found:
                        found[path] = None
                        queue.append(path.read_text())
    return sorted((p for p in found if p not in exclude), key=lambda p: (p.parent, p.name != "__init__.py", p.name))


def shipped_path(file: Path) -> str:
    """Where `file` lands next to the notebook, i.e. its path relative to whatever is on sys.path."""
    for package in settings.PACKAGES.values():
        if file.is_relative_to(package):
            return file.relative_to(package.parent).as_posix()
    return file.relative_to(settings.SOLUTION_DIR).as_posix()


def module_cells(files: list[Path]) -> list[dict]:
    """`%%writefile` cells that recreate `files`, preceded by a cell creating their directories."""
    paths = [shipped_path(f) for f in files]
    directories = sorted({Path(p).parent.as_posix() for p in paths} - {"."})
    mkdir = "import os\n\n" + "\n".join(f'os.makedirs("{d}", exist_ok=True)' for d in directories)
    cells = [make_cell("code", mkdir)] if directories else []
    for file, path in zip(files, paths):
        cells.append(make_cell("code", f"%%writefile {path}\n{file.read_text()}"))
    return cells


def build(entry: Path, user: str | None = None) -> Build:
    """Generate build/<name>/; the notebook belongs to `user`, by default whoever the kaggle CLI is signed in as."""
    entry = entry.resolve()
    name = entry_name(entry)
    source = entry.read_text()
    options = parse_options(source)
    metadata = kernel_metadata(entry, options, user or kaggle_cli.current_user())

    files = imported_files(source, exclude=(entry,))
    note = (
        f"> Generated by `submission_pipeline/notebook.py` from `{entry.relative_to(settings.ROOT).as_posix()}` and the "
        "modules it imports from `solution/` and `models/`. Edit those, not this notebook: the `%%writefile` cells are copies."
    )
    section = make_cell("markdown", "## Modules\n\nThe code this notebook imports, written to disk first.")
    entry_cells = percent_cells(source)
    # keep the entry's own title on top, then the note, then the package, then the rest
    head = entry_cells[:1] if entry_cells and entry_cells[0]["cell_type"] == "markdown" else []
    cells = [*head, make_cell("markdown", note), section, *module_cells(files), *entry_cells[len(head) :]]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    directory = settings.BUILD_DIR / name
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    (directory / metadata["code_file"]).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return Build(name, directory, metadata["id"], options)
