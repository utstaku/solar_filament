import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from synthetic import SIZE

from scoring import rle
from scoring.check import check
from submission_pipeline import kaggle_cli, notebook, settings

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
    assert notebook.entry_name(settings.ROOT / "error_analysis" / "cnn_errors.py") == "error-analysis-cnn-errors"
    with pytest.raises(SystemExit):
        notebook.entry_name(settings.ROOT / "tests" / "synthetic.py")


def test_kernel_metadata_is_derived_from_the_entry_path():
    entry = settings.SOLUTION_DIR / "unet_tiles" / "train.py"
    options = notebook.KernelOptions(gpu=True, kernel_sources=["{user}/filament-unet-tiles-train", "other/thing"])
    metadata = notebook.kernel_metadata(entry, options, "someone")
    assert metadata["id"] == "someone/filament-unet-tiles-train"
    assert metadata["title"] == "Filament Unet Tiles Train"  # slugifies back to the id, as Kaggle requires
    assert metadata["code_file"] == "filament-unet-tiles-train.ipynb"
    assert metadata["enable_gpu"] and metadata["is_private"]
    assert metadata["competition_sources"] == [settings.COMPETITION]
    assert metadata["kernel_sources"] == ["someone/filament-unet-tiles-train", "other/thing"]  # {user} filled in


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

    shared = {
        "dataset/__init__.py",
        "dataset/annotations.py",
        "dataset/paths.py",
        "scoring/__init__.py",
        "scoring/metric.py",
        "scoring/rle.py",  # via submission
        "scoring/submission.py",  # never scoring/check.py: that runs locally
        "preprocess/__init__.py",
        "preprocess/disk.py",
        "preprocess/images.py",
    }
    approach = {"classical_baseline/__init__.py", "classical_baseline/detector.py"}  # never main.py, the entry itself
    assert set(written) == shared | approach
    for relative, body in written.items():
        source = (
            settings.ROOT / relative
            if relative.split("/")[0] in settings.PACKAGES
            else settings.SOLUTION_DIR / relative
        )
        assert body.strip("\n") == source.read_text().strip("\n")
        target = tmp_path / "kaggle" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)

    probe = "import scoring.submission, dataset.annotations, preprocess.disk, classical_baseline.detector; print(scoring.__file__)"
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=tmp_path / "kaggle", capture_output=True, text=True, check=True
    )
    assert result.stdout.strip().startswith(str(tmp_path.resolve()))  # the copy, not the installed package


def test_cnn_entry_ships_its_model_and_approach_modules(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "BUILD_DIR", tmp_path / "build")
    build = notebook.build(settings.SOLUTION_DIR / "cnn_baseline" / "main.py", user="someone")
    assert build.kernel_id == "someone/filament-cnn-baseline"
    metadata = json.loads((build.directory / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is True  # from the "# kaggle: gpu=true" line
    cells = json.loads((build.directory / metadata["code_file"]).read_text())["cells"]
    written = {
        "".join(c["source"]).split("\n", 1)[0].split(" ", 1)[1]
        for c in cells
        if "".join(c["source"]).startswith("%%writefile ")
    }
    assert {"models/__init__.py", "models/unet.py", "cnn_baseline/data.py", "cnn_baseline/training.py"} <= written
    assert not any(name.endswith("main.py") for name in written) and "classical_baseline/detector.py" not in written
    models = settings.PACKAGES["models"]
    assert notebook.resolve("models.unet") == [models / "__init__.py", models / "unet.py"]
    assert notebook.shipped_path(models / "unet.py") == "models/unet.py"  # importable as models.unet


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
        "../zoo/__init__.py": "",  # a package outside solution/, like models/
        "../zoo/net.py": "from common.metric import PanopticQuality",
        "../zoo/unused.py": "",
    }
    for name, body in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(body)
    monkeypatch.setattr(settings, "SOLUTION_DIR", tmp_path)
    monkeypatch.setattr(settings, "PACKAGES", {"zoo": (tmp_path / "../zoo").resolve()})

    entry = tmp_path / "unet" / "train.py"
    source = (
        "%matplotlib inline\nimport torch\n"
        + entry.read_text()
        + "from yolo.detector import detect\nfrom zoo.net import Net\n"
    )
    files = notebook.imported_files(source, exclude=(entry,))
    shipped = [notebook.shipped_path(f) for f in files]
    assert shipped == [
        "common/__init__.py",
        "common/metric.py",
        "common/rle.py",
        "unet/__init__.py",
        "unet/model.py",
        "yolo/detector.py",
        "zoo/__init__.py",
        "zoo/net.py",
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


def test_rejected_push_stops_the_run(monkeypatch, tmp_path):
    """`kaggle kernels push` exits 0 even when Kaggle refuses the notebook."""
    url = "https://www.kaggle.com/code/teammate/filament-error-analysis-cnn-errors"
    replies = iter(
        [
            f"Kernel version 3 successfully pushed.  Please check progress at {url}\n",
            f"Kernel push error: You must accept this competition's rules\n{url}\n",
        ]
    )
    monkeypatch.setattr(kaggle_cli, "kaggle", lambda *args, **kwargs: next(replies))
    kaggle_cli.push(tmp_path)
    with pytest.raises(SystemExit):
        kaggle_cli.push(tmp_path)


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
