import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from synthetic import SIZE

from common import rle
from submission_pipeline import kaggle_cli, notebook, settings
from submission_pipeline.check import check

ENTRY = settings.SOLUTION_DIR / "classical_baseline" / "main.py"


def test_percent_cells_and_options():
    source = (
        "# kaggle: gpu=true kernel_sources=someone/weights,someone/more\n"
        "# %% [markdown]\n# # Title\n#\n# text\n\n# %%\nimport os\n\nx = 1\n\n# %%\n\n# %% [markdown]\n# ## End\n"
    )
    cells = notebook.percent_cells(source)
    assert [c["cell_type"] for c in cells] == ["markdown", "code", "markdown"]  # empty cells are dropped
    assert "".join(cells[0]["source"]) == "# Title\n\ntext"
    assert "".join(cells[1]["source"]) == "import os\n\nx = 1"
    assert "kaggle:" not in json.dumps(cells)  # the options line never reaches the notebook

    options = notebook.parse_options(source)
    assert options.gpu and options.internet and options.kernel_sources == ["someone/weights", "someone/more"]
    assert not notebook.parse_options("# %%\nx = 1\n").gpu
    with pytest.raises(SystemExit):
        notebook.parse_options("# kaggle: gpus=true\n")


def test_entry_name_comes_from_the_path_under_solution():
    solution = settings.SOLUTION_DIR
    assert notebook.entry_name(solution / "unet_tiles" / "train.py") == "unet-tiles-train"
    assert notebook.entry_name(solution / "unet_tiles" / "main.py") == "unet-tiles"  # main.py = the directory itself
    assert notebook.entry_name(solution / "main.py") == "main"
    with pytest.raises(SystemExit):
        notebook.entry_name(settings.ROOT / "tests" / "synthetic.py")


def test_kernel_metadata_is_derived_from_the_entry_path():
    entry = settings.SOLUTION_DIR / "unet_tiles" / "train.py"
    metadata = notebook.kernel_metadata(entry, notebook.KernelOptions(gpu=True), "someone")
    assert metadata["id"] == "someone/filament-unet-tiles-train"
    assert metadata["title"] == "Filament Unet Tiles Train"  # slugifies back to the id, as Kaggle requires
    assert metadata["code_file"] == "filament-unet-tiles-train.ipynb"
    assert metadata["enable_gpu"] and metadata["is_private"]
    assert metadata["competition_sources"] == [settings.COMPETITION]


def test_generated_notebook_is_self_contained(tmp_path, monkeypatch):
    """The modules written by the %%writefile cells must import on their own and match solution/ exactly."""
    monkeypatch.setattr(settings, "BUILD_DIR", tmp_path / "build")
    build = notebook.build(ENTRY, user="someone")  # no kaggle sign-in needed to run the tests
    metadata = json.loads((build.directory / "kernel-metadata.json").read_text())
    assert build.name == "classical-baseline" and build.directory == tmp_path / "build" / "classical-baseline"
    assert build.kernel_id == metadata["id"] == "someone/filament-classical-baseline"
    cells = json.loads((build.directory / metadata["code_file"]).read_text())["cells"]
    assert cells[0]["cell_type"] == "markdown"  # the entry's own title stays on top

    written = {}
    for cell in cells:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code":
            continue
        if source.startswith("%%writefile "):
            header, _, body = source.partition("\n")
            written[header.split(" ", 1)[1]] = body
        else:
            compile(source, "<cell>", "exec")  # every ordinary code cell is valid Python

    parent = settings.SOLUTION_DIR
    common = {"__init__", "annotations", "metric", "paths", "preprocess", "rle", "submission"}  # rle via submission
    approach = {"classical_baseline/__init__.py", "classical_baseline/detector.py"}  # never main.py, the entry itself
    assert set(written) == {f"common/{name}.py" for name in common} | approach
    for relative, body in written.items():
        assert body.strip("\n") == (parent / relative).read_text().strip("\n")
        target = tmp_path / "kaggle" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)

    probe = "import common.submission, classical_baseline.detector; print(common.__file__)"
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=tmp_path / "kaggle", capture_output=True, text=True, check=True
    )
    assert result.stdout.strip().startswith(str(tmp_path.resolve()))  # the copy, not the installed package


def test_only_imported_modules_are_shipped(tmp_path, monkeypatch):
    """Follow imports through solution/: unused modules, other approaches and the entry itself stay out."""
    files = {
        "common/__init__.py": "",
        "common/rle.py": "import numpy",
        "common/metric.py": "from common import rle",
        "common/unused.py": "",
        "unet/__init__.py": "",
        "unet/model.py": "from common.metric import PanopticQuality",
        "unet/train.py": "from unet.model import Net\n",  # the entry
        "unet/predict.py": "from unet.model import Net\n",  # another entry of the same approach
        "yolo/detector.py": "",  # a directory without __init__.py is still a package
        "yolo/unused.py": "",
    }
    for name, body in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(body)
    monkeypatch.setattr(settings, "SOLUTION_DIR", tmp_path)

    entry = tmp_path / "unet" / "train.py"
    source = "%matplotlib inline\nimport torch\n" + entry.read_text() + "from yolo.detector import detect\n"
    shipped = [f.relative_to(tmp_path).as_posix() for f in notebook.imported_files(source, exclude=(entry,))]
    assert shipped == [
        "common/__init__.py",
        "common/metric.py",
        "common/rle.py",
        "unet/__init__.py",
        "unet/model.py",
        "yolo/detector.py",
    ]
    assert notebook.imported_files("import torch\n") == [] and notebook.module_cells([]) == []


def test_current_user_is_read_from_the_kaggle_config(monkeypatch):
    replies = iter(
        ["Configuration values from ~/.kaggle\n- username: teammate\n- auth_method: OAUTH\n", "- username: None\n"]
    )
    monkeypatch.setattr(kaggle_cli, "kaggle", lambda *args, **kwargs: next(replies))
    assert kaggle_cli.current_user() == "teammate"
    with pytest.raises(SystemExit):
        kaggle_cli.current_user()


def test_wait_survives_network_errors(monkeypatch):
    """A traceback mentioning 'Error' is a failed poll, not a failed notebook."""
    traceback = "requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.kaggle.com'): Max retries exceeded"
    running = 'someone/nb has status "KernelWorkerStatus.RUNNING"'

    def replay(*responses):
        queue = list(responses)
        monkeypatch.setattr(kaggle_cli, "kaggle", lambda *args, **kwargs: queue.pop(0))

    assert kaggle_cli.parse_status(traceback) is None
    replay(running, traceback, traceback, 'someone/nb has status "KernelWorkerStatus.COMPLETE"')
    assert kaggle_cli.wait("someone/nb", poll_sec=0) is True
    replay(running, 'someone/nb has status "KernelWorkerStatus.ERROR"')
    assert kaggle_cli.wait("someone/nb", poll_sec=0) is False
    replay(*[traceback] * 3)
    with pytest.raises(SystemExit):
        kaggle_cli.wait("someone/nb", poll_sec=0, max_hiccups=3)


def test_check_accepts_good_and_flags_bad_submissions(tmp_path):
    mask = np.zeros((SIZE, SIZE), np.uint8)
    mask[600:640, 500:700] = 1
    good = pd.DataFrame(
        {"filament_id": ["20110120105534Ch_1", "20110130110334Ch_1"], "segmentation_rle": [rle.encode(mask)] * 2}
    )
    good.to_csv(tmp_path / "good.csv", index=False)
    assert check(tmp_path / "good.csv") == []

    bad = pd.concat([good, good.iloc[[0]]], ignore_index=True)  # duplicated id
    bad.loc[1, "segmentation_rle"] = "not-an-rle"
    bad.loc[1, "filament_id"] = "foo_1"
    bad.to_csv(tmp_path / "bad.csv", index=False)
    errors = " | ".join(check(tmp_path / "bad.csv"))
    assert "duplicated" in errors and "malformed" in errors and "invalid RLE" in errors

    good.rename(columns={"segmentation_rle": "rle"}).to_csv(tmp_path / "columns.csv", index=False)
    assert "columns must be" in check(tmp_path / "columns.csv")[0]
