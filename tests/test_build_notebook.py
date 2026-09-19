import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_notebook


def test_percent_cells():
    source = "# %% [markdown]\n# # Title\n#\n# text\n\n# %%\nimport os\n\nx = 1\n\n# %%\n\n# %% [markdown]\n# ## End\n"
    cells = build_notebook.percent_cells(source)
    assert [c["cell_type"] for c in cells] == ["markdown", "code", "markdown"]  # the empty cell is dropped
    assert "".join(cells[0]["source"]) == "# Title\n\ntext"
    assert "".join(cells[1]["source"]) == "import os\n\nx = 1"


def test_generated_notebook_is_self_contained(tmp_path):
    """The package written by the %%writefile cells must import on its own and match src/ exactly."""
    out_dir = build_notebook.build(ROOT / "kernels" / "classical-baseline")
    metadata = json.loads((out_dir / "kernel-metadata.json").read_text())
    notebook = json.loads((out_dir / metadata["code_file"]).read_text())
    assert notebook["cells"][0]["cell_type"] == "markdown"  # the kernel's own title stays on top

    written = {}
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code":
            continue
        if source.startswith("%%writefile "):
            header, _, body = source.partition("\n")
            written[header.split(" ", 1)[1]] = body
        else:
            compile(source, "<cell>", "exec")  # every ordinary code cell is valid Python

    package = ROOT / "src"
    assert set(written) == {p.relative_to(package).as_posix() for p in (package / "solar_filament").rglob("*.py")}
    for relative, body in written.items():
        assert body.strip("\n") == (package / relative).read_text().strip("\n")
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)

    check = (
        "import solar_filament.classical, solar_filament.submission, solar_filament.annotations, "
        "solar_filament.paths; print(solar_filament.__file__)"
    )
    result = subprocess.run([sys.executable, "-c", check], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout.strip().startswith(str(tmp_path.resolve()))  # imported the copy, not the installed package
