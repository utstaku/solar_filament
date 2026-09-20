# Solar Filament Segmentation Challenge 2026

Kaggle: <https://www.kaggle.com/competitions/filament-segmentation-2026>（締切 2026-11-15 / IEEE BigData Cup 2026）

GONG Hα全球画像（2048×2048, 8bitグレースケール）からフィラメントを**個体ごと**にセグメンテーションする。
評価は Panoptic Quality（IoU > 0.5 でTP）。提出は1行=1フィラメントのCSV（`filament_id`, `segmentation_rle`）。
コンペ仕様・データの知見・アーキテクチャ調査は [docs/research.md](docs/research.md)。

## リポジトリは2つの場所でできている

```
solution/     ★ 開発はここだけ。中身は純粋なPython
  common/                  どの手法でも使う共通処理だけ: データの場所・正解の読込・評価指標(PQ)・RLE・提出CSV・汎用の前処理
  classical_baseline.py    実行スクリプト（1ファイル = Kaggle notebook 1本）。その手法だけのコードはここに書く
  <新しい実験>.py           例: unet_tiles.py → notebook「filament-unet-tiles」になる
  <手法名>/                 手法専用のコードが1ファイルに収まらなくなったら作る（例: unet/model.py, unet/train.py）

pipeline/     ★ 提出パイプライン一式。普段は触らない。コマンドは1つ
  cli.py  notebook.py  kaggle_cli.py  check.py  settings.py

tests/        合成の全球画像でのテスト（実データ不要）
docs/         調査メモ
outputs/ build/   取得した成果物 / 生成されたnotebook（どちらもgit管理外）
```

## パイプライン（1コマンド）

```bash
uv run filament run solution/classical_baseline.py                  # 形式チェックまで
uv run filament run solution/classical_baseline.py --submit "メモ"   # さらに提出（確認プロンプトあり）
```

`run` が順に行うこと:

1. `pytest`（失敗したらpushしない。GPU枠をバグで溶かさないため）
2. `solution/<entry>.py` と、それが import している `solution/` 内のモジュール**だけ**から自己完結したnotebookを生成（`build/<entry>/`）
3. Kaggleへpush（private）→ 完了まで待機
4. ログと小さい出力（csv/json/txt）だけを `outputs/<entry>/` に取得。重みや画像はKaggleに残す
5. `submission.csv` の形式チェック（列・ID・RLEの妥当性・空マスク・重なり）
6. `--submit` 指定時のみ: 残り回数を表示 → `[y/N]` 確認 → 提出 → 結果表示（1日5回まで）

```bash
uv run filament submit outputs/classical_baseline/submission.csv "メモ"   # 取得済みCSVを提出
uv run filament check  outputs/classical_baseline/submission.csv          # 形式チェックだけ
uv run filament run solution/x.py --build-only                            # notebook生成だけ
```

## 開発の流れ

1. **書く**: 手順とその手法だけのコードは `solution/<実験名>.py`（大きくなったら `solution/<手法名>/`）、
   どの手法でも使う処理だけ `solution/common/`。実行スクリプトは percent 形式（`# %%` でコードセル、
   `# %% [markdown]` でMarkdownセル）で、`from common.metric import ...` と普通に import する。
2. **ローカルで確かめる**: `uv run pytest`。データは無いので `tests/synthetic.py` の合成画像
   （灰色の空・明るいリム・周縁減光・低コントラストのフィラメント）とダミーテンソルで、形状・損失・数ステップの学習まで。
3. **Kaggleで実行**: `uv run filament run solution/<実験名>.py`。図はnotebook内に描画されるのでKaggleのページで見る。
4. **結果を見る**: `outputs/<実験名>/run.log` とレポートJSON。直して 1 に戻る。

Kaggle側の設定ファイルは書かない。notebookのslugとタイトルはファイル名から決まり、実行オプションは
実行スクリプト先頭のコメントで指定する:

```python
# kaggle: gpu=true accelerator=NvidiaTeslaT4 kernel_sources=takuroabe/filament-unet-train
```

（`gpu` / `internet` / `accelerator` / `kernel_sources` / `dataset_sources` / `model_sources`。GPUは週30h枠。
学習済み重みは学習notebookの出力としてKaggleに残し、推論側から `kernel_sources` で参照する。）

生成されたnotebookでは、実行スクリプトが（間接的にでも）import しているモジュールだけが `%%writefile` セルになりKaggle上で実ファイルとして
書き出されるので、import もトレースバックの行番号もローカルと同じ。生成物（`build/`）は手で編集しない・
gitに入れない。Kaggle上で試行錯誤した変更は `solution/` に書き戻す。最終提出用のnotebookも同じ仕組みで生成する。

## 方針：データはローカルに置かない

コンペデータはダウンロードしない。実行はすべて Kaggle Notebooks 上。ローカルに戻すのはログ・提出CSV・
レポート等の小さな成果物だけ。データを覗きたいときは小さなnotebookを流してログや図で見る。

## セットアップ

```bash
brew install uv          # 未導入の場合
uv sync                  # .venv を作成（uv管理の Python 3.12）。common と filament コマンドも入る
uv run kaggle --version  # 認証は ~/.kaggle/credentials.json（`kaggle auth login`）を利用
uv run ruff check . && uv run ruff format .
```

コンペ名・Kaggleユーザー名などは [pipeline/settings.py](pipeline/settings.py)。

## 最終提出の要件（11/15まで）

公開Gitリポジトリ・`requirements.txt`（`uv export --no-hashes --no-dev -o requirements.txt`）・
パイプライン全体を示すJupyter notebook・4ページの技術レポート。コード品質も採点対象（定性評価30%）。
リポジトリの公開は締切後（開催中のチーム外へのコード共有はKaggleのルールで制限される）。
