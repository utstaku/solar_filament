# Solar Filament Segmentation Challenge 2026

Kaggle: <https://www.kaggle.com/competitions/filament-segmentation-2026>（締切 2026-11-15 / IEEE BigData Cup 2026）

GONG Hα全球画像（2048×2048, 8bitグレースケール）からフィラメントを**個体ごと**にセグメンテーションする。
評価は Panoptic Quality（IoU > 0.5 でTP）。提出は1行=1フィラメントのCSV（`filament_id`, `segmentation_rle`）。
コンペ仕様・データの知見・アーキテクチャ調査は [docs/research.md](docs/research.md)。

## リポジトリの構成

```
dataset/      ★ 配布データ（手法に依らず固定）
  paths.py  annotations.py     データの場所 / 正解JSONの読込（アノテーターごとの注釈を画像単位にまとめる）

scoring/      ★ 採点のされ方（手法に依らず固定）
  metric.py  rle.py  submission.py  check.py
              Panoptic Quality / マスク⇄RLE / 提出CSVの書き出し / 提出CSVの形式チェック（ローカル専用）

preprocess/   ★ 画像の前処理（手法をまたいで使い回す）
  images.py  disk.py        JPEG読込 / 円盤検出・周縁減光の除去

models/       ★ モデル定義（学習の手順は含めない）。`from models.unet import UNet`
  unet.py                  ごく普通のU-Net（事前学習なし、pure PyTorch）

solution/     ★ 手法ごとのディレクトリ。開発の中心
  classical_baseline/      手法1つ = ディレクトリ1つ
    detector.py              その手法だけのコード（main.py から import）
    main.py                  実行スクリプト（1ファイル = Kaggle notebook 1本）→ notebook「filament-classical-baseline」
  cnn_baseline/            CNNによる最初の学習型ベースライン
    data.py                  読込・ソフトラベル（複数アノテーターの平均）・タイル切り出し
    training.py              学習ループ（AdamW + one-cycle、CUDAではAMP）と全フレーム推論
    postprocess.py           確率マップ → 個体（しきい値 + 連結成分）
    main.py                  → notebook「filament-cnn-baseline」（GPU）
  <次の手法>/               学習用と推論用を分けるなら train.py / predict.py → 「filament-<手法>-train」「…-predict」

submission_pipeline/   ★ ローカルでKaggleを操作する道具。Kaggleには送られない。コマンドは1つ
  cli.py  notebook.py  kaggle_cli.py  settings.py

tests/        合成の全球画像でのテスト（実データ不要）
docs/         調査メモ
outputs/ build/   取得した成果物 / 生成されたnotebook（どちらもgit管理外）
```

## パイプライン（1コマンド）

```bash
uv run filament run solution/classical_baseline/main.py                  # 形式チェックまで
uv run filament run solution/classical_baseline/main.py --submit "メモ"   # さらに提出（確認プロンプトあり）
```

`run` が順に行うこと:

1. `pytest`（失敗したらpushしない。GPU枠をバグで溶かさないため）
2. 実行スクリプトと、それが import しているモジュール（`solution/`・`dataset/`・`scoring/`・`preprocess/`・`models/`）**だけ**から自己完結したnotebookを生成（`build/<name>/`）
3. Kaggleへpush（private）→ 完了まで待機
4. ログと小さい出力（csv/json/txt）だけを `outputs/<name>/` に取得。重みや画像はKaggleに残す
5. `submission.csv` の形式チェック（列・ID・RLEの妥当性・空マスク・重なり）
6. `--submit` 指定時のみ: 残り回数を表示 → `[y/N]` 確認 → 提出 → 結果表示（1日5回まで）

```bash
uv run filament submit outputs/classical-baseline/submission.csv "メモ"   # 取得済みCSVを提出
uv run filament check  outputs/classical-baseline/submission.csv          # 形式チェックだけ
uv run filament run solution/unet/train.py --build-only                   # notebook生成だけ
```

`<name>` は実行スクリプトの `solution/` からの相対パス（`unet/train.py` → `unet-train`）。
`main.py` だけはディレクトリ名そのもの（`classical_baseline/main.py` → `classical-baseline`）。
notebook名はこれに `filament-` を付けたもの。

## 開発の流れ

1. **書く**: 手法ごとに `solution/<手法名>/` を作り、その手法だけのコード（損失・学習ループ・後処理）と
   実行スクリプト（`main.py`、または `train.py` / `predict.py` のように複数）を置く。モデルの定義は `models/`、
   手法をまたぐ前処理は `preprocess/`、配布データと採点の形式は `dataset/`・`scoring/`（普段は変えない）。
   実行スクリプトは percent 形式（`# %%` でコードセル、`# %% [markdown]` でMarkdownセル）で、
   `from scoring.metric import ...` や `from unet.model import ...` と普通に import する（`__init__.py` は無くてもよい）。
2. **ローカルで確かめる**: `uv run pytest`。データは無いので `tests/synthetic.py` の合成画像
   （灰色の空・明るいリム・周縁減光・低コントラストのフィラメント）とダミーテンソルで、形状・損失・数ステップの学習まで。
   notebook全体の流れも、合成データをKaggleと同じ配置で置き、`FILAMENT_DATA=<その場所> FILAMENT_OUTPUT=<出力先>`
   を付けて実行スクリプトを `python` で走らせれば、GPUを使う前にローカルで通せる
   （macOSでは DataLoader の `num_workers=0` にすること。spawn方式のワーカーがスクリプト全体を再実行してしまう）。
3. **Kaggleで実行**: `uv run filament run solution/<手法名>/<スクリプト>.py`。図はnotebook内に描画されるのでKaggleのページで見る。
4. **結果を見る**: `outputs/<name>/run.log` とレポートJSON。直して 1 に戻る。

Kaggle側の設定ファイルは書かない。notebookのslugとタイトルはスクリプトのパスから決まり、実行オプションは
実行スクリプト先頭のコメントで指定する:

```python
# kaggle: gpu=true accelerator=NvidiaTeslaT4 kernel_sources=<Kaggleユーザー名>/filament-unet-train
```

（`gpu` / `internet` / `accelerator` / `kernel_sources` / `dataset_sources` / `model_sources`。GPUは週30h枠。
学習済み重みは学習notebookの出力としてKaggleに残し、推論側から `kernel_sources` で参照する。）

生成されたnotebookでは、実行スクリプトが（間接的にでも）import しているモジュールだけが `%%writefile` セルになりKaggle上で実ファイルとして
書き出されるので、import もトレースバックの行番号もローカルと同じ。生成物（`build/`）は手で編集しない・
gitに入れない。Kaggle上で試行錯誤した変更は `solution/` に書き戻す。最終提出用のnotebookも同じ仕組みで生成する。

## 方針：データはローカルに置かない

コンペデータはダウンロードしない。実行はすべて Kaggle Notebooks 上。ローカルに戻すのはログ・提出CSV・
レポート等の小さな成果物だけ。データを覗きたいときは小さなnotebookを流してログや図で見る。

## セットアップ（チームメンバー共通）

```bash
git clone git@github.com:utstaku/solar_filament.git && cd solar_filament
brew install uv          # 未導入の場合（Linux: curl -LsSf https://astral.sh/uv/install.sh | sh）
uv sync                  # .venv を作成（uv管理の Python 3.12）。各パッケージと filament コマンドも入る
uv run kaggle auth login # 自分のKaggleアカウントでログイン（認証情報は ~/.kaggle に置かれ、リポジトリには入らない）
uv run pytest            # 動作確認（データもネットワークも不要）
```

そのうえで Kaggle 側で (1) コンペに参加（ルールに同意。これがないとnotebookにデータを付けられない）、
(2) チームをマージ（Team タブ。提出は**チーム合計で**1日5回）。

各自の環境で `uv run filament run ...` を実行すると、notebookは**ログイン中の自分のアカウント**の下に
private で作られる（ユーザー名は `kaggle config view` から自動で取る。設定ファイルの編集は不要）。
他のメンバーのnotebookの出力（学習済み重みなど）を `kernel_sources` で参照したいときは、
そのnotebookの共有設定で相手を collaborator に追加してもらう。

コンペ名などは [submission_pipeline/settings.py](submission_pipeline/settings.py)。動作確認済みは macOS（Apple Silicon）。
Linux でも同じ手順で動く見込み。Windows は未確認。

## 最終提出の要件（11/15まで）

公開Gitリポジトリ・`requirements.txt`（`uv export --no-hashes --no-dev -o requirements.txt`）・
パイプライン全体を示すJupyter notebook・4ページの技術レポート。コード品質も採点対象（定性評価30%）。
リポジトリの公開は締切後（開催中のチーム外へのコード共有はKaggleのルールで制限される）。
