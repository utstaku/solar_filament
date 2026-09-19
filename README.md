# Solar Filament Segmentation Challenge 2026

Kaggle: <https://www.kaggle.com/competitions/filament-segmentation-2026>（締切 2026-11-15 / IEEE BigData Cup 2026）

GONG Hα全球画像（2048×2048, 8bitグレースケール）からフィラメントを**個体ごと**にセグメンテーションする。
評価は Panoptic Quality（IoU > 0.5 でTP）。提出は1行=1フィラメントのCSV（`filament_id`, `segmentation_rle`）。

## 方針：データはローカルに置かない

コンペデータはダウンロードしない。コードをローカルで書き、実行はすべて Kaggle Notebooks 上で行う。
ローカルに戻すのはログ・提出CSV・レポート等の小さな成果物だけ（`outputs/`、git管理外）。

```
ローカル (uv)                                          Kaggle
src/solar_filament/*.py ─┐
kernels/<name>/main.py  ─┴ build ─▶ build/<name>/*.ipynb ── push ──▶ private notebook（データは /kaggle/input）
outputs/<name>/*.csv    ◀──────────────────────────────── output ─  /kaggle/working/submission.csv
scripts/submit.sh       ───────────────────────────────── submit ─▶ leaderboard
```

## 開発の流れ

1. **書く**: 共通コードは `src/solar_filament/`、実行手順は `kernels/<name>/main.py`。
   main.py は percent 形式（`# %%` でコードセル、`# %% [markdown]` でMarkdownセル）で書き、
   `from solar_filament.metric import ...` のように普通に import する。
2. **ローカルで確かめる**: `uv run pytest`。実データは無いので `tests/synthetic.py` の合成全球画像を使う
   （灰色の空・明るいリム・周縁減光・低コントラストのフィラメントを再現）。
3. **Kaggleで実行**: `scripts/kernel_run.sh kernels/<name>`。`scripts/build_notebook.py` が
   自己完結したnotebookを生成してからpushする。パッケージの各モジュールは `%%writefile` セルになり、
   Kaggle上で実ファイルとして書き出されるので、import もトレースバックの行番号もローカルと同じ。
   図はnotebook内に描画されるのでKaggleのページで見る（画像はローカルに落とさない）。
4. **結果を見る**: `outputs/<name>/run.log` とレポートJSON。直して 1 に戻る。

生成した `.ipynb`（`build/`）は成果物なので手で編集しない・gitに入れない。Kaggle上で試行錯誤した変更は
`src/` か main.py に書き戻す。最終提出用のnotebookも同じ仕組みで生成する。

## セットアップ

```bash
brew install uv          # 未導入の場合
uv sync                  # .venv を作成（Python 3.12）
uv run kaggle --version  # 認証は ~/.kaggle/credentials.json（`kaggle auth login`）を利用
```

## 使い方

```bash
uv run pytest                                   # 合成データでのテスト
uv run ruff check . && uv run ruff format .     # lint / format

# notebook生成 → push → 完了待ち → ログと小さい出力（csv/json/txt）を outputs/<name>/ に取得
scripts/kernel_run.sh kernels/classical-baseline
uv run python scripts/build_notebook.py kernels/classical-baseline      # 生成だけして中身を確認したいとき
scripts/kernel_run.sh kernels/<gpu-kernel> --accelerator NvidiaTeslaT4   # GPUは週30h枠

# 提出（形式チェック → 残り回数表示 → 確認プロンプト → submit）。1日5回まで
scripts/submit.sh --check outputs/classical-baseline/submission.csv     # チェックのみ
scripts/submit.sh outputs/classical-baseline/submission.csv "message"

# その他
uv run kaggle competitions submissions filament-segmentation-2026
uv run kaggle competitions leaderboard filament-segmentation-2026 --show
uv run kaggle competitions pages filament-segmentation-2026 list filament-segmentation-2026 --content
```

学習済み重みは kernel の出力として Kaggle 上に残し、推論 kernel からは `kernel-metadata.json` の
`kernel_sources` で参照する（ローカルには落とさない。`OUTPUT_PATTERN` を変えない限りダウンロード対象外）。

## 構成

| パス | 内容 |
| --- | --- |
| `src/solar_filament/` | 共通パッケージ: アノテーション読込、RLE、Panoptic Quality、前処理（円盤検出・周縁減光除去）、古典的検出、提出CSV |
| `kernels/<name>/` | Kaggleで実行する手順（percent形式の `main.py` + `kernel-metadata.json`） |
| `scripts/` | notebook生成、push/待機/取得、提出、提出CSVの形式チェック |
| `tests/` | 合成データによるテスト（PQは pycocotools のIoUによる参照実装と照合） |
| `build/` | 生成されたnotebook（git管理外） |
| `docs/research.md` | コンペ仕様の要点とアーキテクチャ調査 |
| `outputs/` | Kaggleから取得した成果物（git管理外） |

## 最終提出の要件（11/15まで）

公開Gitリポジトリ・`requirements.txt`（`uv export --no-hashes --no-dev -o requirements.txt`）・
パイプライン全体を示すJupyter notebook・4ページの技術レポート。コード品質も採点対象（定性評価30%）。
