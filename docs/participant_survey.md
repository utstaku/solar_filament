# 他参加者の公開手法サーベイ（2026-10-04）

## 結論と今回の実験の位置づけ

「既存U-Netを超えられない」ではなく、「今回の16epoch・単一分割・設定では超えなかった」が正確。
同じヒステリシス＋OOF成分フィルタで、既存U-Net 0.3767、EfficientNet-B3 0.3732、ConvNeXt-Tiny 0.3520、EfficientNet＋Skeleton Recall 0.3615だった。
既存と新モデルには推論方式、円盤外処理、学習率、バッチサイズの差もある。0.0035差の統計的有意性は未確認。

他参加者にもU-Net系が多いが、個体提案器、境界損失、複数アノテーターの扱い、偽陽性除去に工夫がある。
次の低コスト実験は既存U-Netへの境界重み付きBCE、構成を大きく変える候補はYOLO検出＋seed付きcrop U-Netとする。
これは本調査からの判断で、以下の公開スコアを自分たちのCVに直接換算したものではない。

## 調査方法・スコアの読み方

- Kaggle APIでコンペの公開notebookを得票順に15件確認。
- Tim Gluhkikh、Nomannic、Lâm Huyの3件は公開notebookのソースを取得し、実行せず読んだ。
- 参加者のGitHub README・実験記録、本人の解説ページを確認。
- 下記のスコアはすべて著者の公開報告。こちらで再学習・再現した値ではなく、Public LBも独立照合していない。
- 単一fold、日付/月単位fold、複数fold平均、LBは比較条件が異なる。高い数値だけで手法を順位付けしない。

## 公開手法

| 公開者 | 手法 | 報告値 | 確認の範囲・留意点 |
| --- | --- | --- | --- |
| Tim Gluhkikh | YOLO11s検出 → 原画像/CLAHE/箱seedのcrop U-Net | ローカルPQ約0.409との記述 | notebookの推論・後処理コードを確認。取得版は保存済み重みを使い学習をスキップ |
| Nomannic | Mask R-CNN ResNet50-FPN-v2 → ResNet18 crop U-Net | 今回はPQの比較値として採用せず | コード確認。refiner選択が`all_dice`基準で、現行PQとの違いに注意 |
| HeShen-1 | YOLO11m-seg＋seed付きResNet18 U-Net | 5-fold平均PQ 0.3593、融合LB 0.35 | README・実験報告。日付単位CV、難しい背景の負例、反転TTA |
| Wzy633 | YOLO26n-seg、重複マスクを排他化 | Fold0 PQ 0.35259、LB 0.31 | 月単位分割。実験記録と訓練スクリプトを確認 |
| bcap52 | ResNet34 U-Net＋境界重み付きBCE | OOF PQ 0.373、LB 0.33 | README報告。参照先の詳細実験台帳は取得できず、独立検証なし |
| Shivam Bhardwaj / too.foo | EfficientNet-B3 U-Net++＋成分選択 | ローカルPQ 0.416、LB 0.32との記述 | 本人の解説ページ。今回、訓練コードの独立確認はしていない |

### 1. 検出 → seed付きcrop U-Net

[Tim Gluhkikhのnotebook](https://www.kaggle.com/code/timgluhkikh/best-score-magfilo-yolo11s-detect-crop-u-net)で確認した構成は、YOLO11sを1280pxで推論し、検出箱の周辺を512pxへリサイズ、`[gray, CLAHE, filled box seed]`から個体マスクを予測するもの。重なりは検出確信度順に排他化する。箱が「どの個体を切り出すか」を指定する点が、今の全画面セマンティック予測と異なる。

取得版の設定は`SKIP_YOLO_TRAIN=True`、`SKIP_REFINER_TRAIN=True`。冒頭ではbox-jitter TTAに言及するが、コードの`USE_BOX_JITTER_TTA`と`USE_TTA`はともにFalse。説明文と実行設定を区別する必要がある。約0.409も記述上の値であり、重みの学習分割との整合性を今回保証したものではない。

[Nomannicのnotebook](https://www.kaggle.com/code/nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner)も個体検出後にcropを精細化する。Mask R-CNNは1024px、cropモデルはImageNet事前学習ResNet18を使う。現行PQで選び直すことを前提に、2段構成の実装例として参照する。

**自分たちへの示唆:** 2段構成の狙いは大きなencoderではなく個体の指定。検出段階の見逃しは後段で救えないので、検出recallと最終PQを分けて測る。最初は外部のコンペ学習済み重みを流用せず、自分のtrain分割で学習する。

### 2. HeShen-1：2段構成の成功・失敗例

[公開リポジトリ](https://github.com/HeShen-1/filament-segmentation-2026)はYOLO11m-seg（1536px、50epoch）＋256pxのseed付きResNet18 U-Netを報告。代表的アノテーターの選択、25%の背景負例、4方向反転TTAが特徴。
50→100epochの延長はfold0では改善したが、paired 5-foldでは平均差−0.0012。単一foldの改善が再現しない具体例として参考になる。

著者はSQが注釈の不一致に制約されると論じるが、人間同士のmatched IoUは数学的な上限ではない。本調査でも「SQ改善は不可能」とは判断しない。

### 3. Wzy633：YOLO-segと評価上の落とし穴

[実験記録](https://github.com/Wzy633/solar-filament-segmentation-2026/blob/main/EXPERIMENTS.md)では1024px・30epoch学習、1536px推論のYOLO26n-seg。月単位のFold0で、重複を許した診断PQ 0.40476が、提出可能な排他マスクでは0.35259になった。重複許容スコアを採用すると大きく過大評価する。

[訓練スクリプト](https://github.com/Wzy633/solar-filament-segmentation-2026/blob/main/scripts/train_detector.py)はモデル名・画像サイズ・epochを引数に取る。既定値だけを最終実験設定と取り違えない。

**示唆:** YOLO-segは比較候補だが、必ず提出と同じ排他化後にPQを測る。報告にある最低1個体を強制する処理などは、自分たちの分割で利益を確認するまで採用しない。

### 4. bcap52：境界を重くする損失

[README](https://github.com/bcap52/solar-filament-segmentation-2026)はResNet34 U-Netの系列で、5-fold ensemble PQ 0.357 → 境界重み付きBCE 0.368 → 境界重み8で0.373と報告。単にencoderを変えるより、細い輪郭の誤差に学習を向ける案。

公開READMEにある`docs/PIPELINE.md`は取得できず、詳細アブレーションの再確認はできなかった。磁場画像の融合も記載されているが、外部観測を推論入力に足す部分は今回の採用案に含めない。

**示唆:** 自分たちの既存U-Netに、配布segmentationから作る境界帯の重み付けを加える。境界の不確実な注釈まで強く学習してしまう可能性があるため、重み2/4程度から小さく比較する（8が最適とは仮定しない）。

### 5. too.foo：EfficientNet系でも後処理を重視

[本人の解説](https://kaggle.too.foo/competitions/filament-segmentation-2026/)はEfficientNet-B3 U-Net++、元解像度のタイル推論、成分選択を報告。ローカルPQ 0.416とPublic 0.32の差も記載される。

「EfficientNetだから勝つ」根拠にはならないが、特徴抽出器の置換だけでなく個体候補の選択まで含めて改善している例。ページ内のLB全体や上限に関する主張は、主催者の確認済み事実として扱わない。

## 高スコアの公開notebookをそのまま根拠にしない理由

[Lâm Huyの「U-Net 0.55+」](https://www.kaggle.com/code/lamhuy8904/solar-filament-unet-segmentation-0-55)の取得コードでは、`CHAMPION_PAYLOAD`をbase85＋lzmaで復元し、CSVとして出力する処理を確認した。その数値は、このnotebook内でU-Netを学習・推論した性能の証拠にはならない。payloadの出所や不正を断定するものではない。

[zlyx2377の構成案](https://github.com/zlyx2377/filament-segmentation-2026)にはConvNeXt/Swin/MiT、spine補助、watershed等が並ぶが、READMEの目標・チェックリストを実測済みの改善として扱わない。外部MAGFiLOの正解利用を示唆する記述も今回の実験案には採用しない。

## 次の実験案（まだ実行しない）

1. **既存U-Net＋境界重み付きBCE**：モデル・分割・入力・後処理を固定し、損失だけ変える。各epochを検証し、最終epochだけではなくPQでcheckpointを選ぶ。ただし選択後は別foldで再確認する。
2. **YOLO11s検出＋seed付きcrop U-Net**：構成の異なる候補を1本作る。既存U-Netの局所的な誤結合・断片化に対し、個体指定が効くか評価。
3. **アノテーター教師の比較**：現在のsoft平均に対し、画像ごとに注釈セットをサンプリングする方法を別実験で比較。平均ラベルとDiceの組合せが単独注釈の薄い個体に与える影響を調べる。

モデル名を増やす前に、同じ分割・推論方式で既存U-Netを再評価する。全体PQに加え、小面積missed、near_miss、fragmented、FPを追う。OOF成分フィルタでも、その閾値を同じ検証データで選べば選択バイアスは残る。

## ローカルの比較記録

- `outputs/pretrained-unet/comparison.json`：今回の3モデル、単一閾値探索。
- `outputs/pretrained-unet-reprocess/reprocess_report.json`：同じヒステリシス＋OOF成分フィルタ。
- `outputs/cnn-baseline-predict/predict_report.json`：既存U-NetのPQ 0.3767とfilter cutoff=0.45の根拠。
