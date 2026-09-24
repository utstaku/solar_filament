"""One command from code to leaderboard.

    uv run filament run solution/classical_baseline/main.py                   # everything up to the format check
    uv run filament run solution/classical_baseline/main.py --submit "note"   # ... and submit (asks first)
    uv run filament submit outputs/classical-baseline/submission.csv "note"
    uv run filament check  outputs/classical-baseline/submission.csv

`run` does, in order: pytest -> build the notebook -> push -> wait -> fetch log and small outputs into
outputs/<name>/ -> check submission.csv -> (with --submit) show the remaining quota, ask, submit.
<name> is the entry's path under solution/, so solution/unet/train.py -> outputs/unet-train/.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from scoring.check import check
from submission_pipeline import kaggle_cli, notebook, settings


def step(title: str) -> None:
    print(f"\n>> {title}", flush=True)


def confirm_and_submit(csv: Path, message: str) -> int:
    if check(csv):
        print("not submitting a file with errors")
        return 1
    print(kaggle_cli.submission_limits())
    try:
        answer = input(f"Submit {csv} to {settings.COMPETITION}? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() != "y":
        print("aborted")
        return 1
    kaggle_cli.submit(csv, message)
    print(kaggle_cli.recent_submissions())
    return 0


def run(args: argparse.Namespace) -> int:
    entry = Path(args.entry)
    if not entry.is_file():
        raise SystemExit(f"{entry} does not exist")

    if not args.skip_tests:
        step("tests")
        pytest = [sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-W", "ignore"]
        if subprocess.run(pytest, check=False).returncode:
            print("tests failed; nothing was pushed (use --skip-tests to override)")
            return 1

    step("build notebook")
    build = notebook.build(entry)
    print(f"{build.directory}  ->  {build.kernel_id}  (gpu={build.options.gpu or bool(build.options.accelerator)})")
    if args.build_only:
        return 0

    step("push")
    kaggle_cli.push(build.directory, build.options.accelerator)
    print(f"https://www.kaggle.com/code/{build.kernel_id}")

    step("wait")
    completed = kaggle_cli.wait(build.kernel_id, args.poll_sec)
    out_dir = settings.OUTPUT_DIR / build.name
    kaggle_cli.fetch_log(build.kernel_id, out_dir / "run.log")
    if not completed:
        print(f"run failed; last lines of {out_dir / 'run.log'}:")
        print("\n".join((out_dir / "run.log").read_text().splitlines()[-40:]))
        return 1

    step(f"fetch outputs -> {out_dir.relative_to(settings.ROOT)}")
    kaggle_cli.fetch_outputs(build.kernel_id, out_dir)
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.stat().st_size / 1024:9.1f} KB  {path.name}")

    csv = out_dir / "submission.csv"
    if not csv.exists():
        print("no submission.csv in the outputs")
        return 1 if args.submit else 0
    step("check submission")
    if args.submit is None:
        return 1 if check(csv) else 0
    return confirm_and_submit(csv, args.submit)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="filament", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("run", help="build, push, wait, fetch, check (and submit with --submit)")
    p.add_argument("entry", help="entry script, e.g. solution/classical_baseline/main.py")
    p.add_argument("--submit", metavar="MESSAGE", help="submit submission.csv afterwards (asks for confirmation)")
    p.add_argument("--build-only", action="store_true", help="stop after generating build/<name>/")
    p.add_argument("--skip-tests", action="store_true", help="do not run pytest first")
    p.add_argument("--poll-sec", type=int, default=30)
    p.set_defaults(func=run)

    p = commands.add_parser("check", help="validate a submission CSV")
    p.add_argument("csv", type=Path)
    p.set_defaults(func=lambda a: 1 if check(a.csv) else 0)

    p = commands.add_parser("submit", help="validate, ask, submit an existing CSV")
    p.add_argument("csv", type=Path)
    p.add_argument("message")
    p.set_defaults(func=lambda a: confirm_and_submit(a.csv, a.message))

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
