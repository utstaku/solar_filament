# 次期モデル調査（2026-10-04）

## 結論

最初に **ImageNet事前学習EfficientNet-B3 + U-Net**、対抗として **ConvNeXt-Tiny + U-Net** を比較する。次に勝者へ細線の連結性を補う損失を追加する。現時点で検出器＋リファイナへの全面移行は優先しない。

これは以下のローカル誤差分析と一次資料からの提案であり、このデータで新モデルの優位性を実測した結果ではない。既存の `research.md` の「2段構成をまず作る」案を、取得済みの誤差に基づいて更新する。

ブランチ起点はローカルmain `179b1b4`。リモートmainとの同期は行っていない。今回の成果は調査と実験設計で、学習・提出は未実施。

## 現状から何を改善するか

mainの `models/unet.py` は事前学習なし、2ch入力、base=32、depth=4。`solution/cnn_baseline/main.py` は512pxタイル、16epoch、BCE+Dice、複数アノテーターの平均マスクで学習する。

参考成果物は `outputs/cnn-baseline-predict/error_report.json` と `breakdowns.txt`。この成果物はmainの再実行結果とは限らず、後処理はthreshold=0.5 / min_area=150 / low=0.2である。新モデルの比較時は同じ分割・後処理でベースラインを再評価すること。

| 観測 | 値 | 示唆（仮説） |
| --- | --- | --- |
| PQ / SQ / RQ | 0.3767 / 0.6657 / 0.5659 | 検出の成否と境界精度の両方に改善余地 |
| GT分類 | missed 377、near_miss 184、fragmented 59、merged 33（全1,387） | 個体分離だけを強化するより、見逃しとIoU不足を先に減らす |
| 最小面積四分位のmissed | 52.6% | 解像度を保つ。ただし面積フィルタの影響も切り分ける |
| 最低コントラスト四分位 | missed 44.4%、near_miss 24.8% | 特徴抽出と境界復元の強化を優先 |
| 最大面積四分位のfragmented | 13.3% | 広い文脈・連結性の補助学習に余地 |
| 円盤周縁のmissed | 52.7% | 前処理・円盤マスクも原因候補。モデルだけとは断定しない |

what-ifのnear misses fixedはPQ=0.4491、no hallucinationsは0.3976。これは仮想的な置換計算で、実現可能な改善量や各施策の加算可能な効果ではない。誤差カテゴリも厳密な因果分解ではない。

## 候補の比較

| 優先 | 候補 | 期待する効果（未検証） | 費用・弱点 |
| --- | --- | --- | --- |
| 1 | EfficientNet-B3 + U-Net | 事前学習特徴とスキップ接続で薄い小個体・境界を改善 | 自然画像からHαへの転移効果は実測が必要 |
| 2 | ConvNeXt-Tiny + U-Net | 広い文脈で背景模様と長いフィラメントを識別 | 標準特徴は1/4解像度から。細い構造に不利なら高解像度の浅い枝を追加 |
| 3 | 勝者 + Skeleton Recall Loss | 大きい個体の断片化を抑える | 個体分離は学習しない。隣接個体の誤接続に注意 |
| 4 | EdgeAttNet系のエッジ誘導attention | 低コントラストの境界・barbを復元 | 同領域の論文だがコンペPQでの優位性は未確認。再実装コストあり |
| 5 | RF-DETR segmentation / Mask2Former + 高解像度refiner | 個体のグルーピングを直接学習 | 注釈セットの扱い、学習基盤、メモリ、細部の復元が追加課題 |

### 事前学習U-Netを最初にする理由

SMPは事前学習エンコーダと任意の入力チャネル数を持つU-Netを提供するため、既存の学習・評価の多くを再利用できる。[SMP公式quickstart](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/docs/quickstart.rst)

ConvNeXt系は通常1/4、1/8、1/16、1/32の特徴を出す。出力を元解像度へ拡大しても失われた細部が自動で戻るわけではないため、EfficientNetとの比較が必要。[SMP公式encoder資料](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/docs/encoders_timm.rst)

初回は入力を既存のraw＋周縁減光補正の2chに固定し、入力層の重み変換と正規化を確認する。3ch化（raw/補正/CLAHE）を同時に変えると事前学習の寄与が分からなくなる。パッケージは既に依存にあるが、選定encoder名・重みの取得方法・実行環境での対応は実装時に確認する。

### 細線向けの追加候補

Skeleton Recall LossはGT骨格を利用して細管構造の連結性を改善し、重いGPU骨格計算を避ける手法。clDiceは予測とGTのマスク・骨格の一致を用いる。いずれも太陽フィラメントPQへの効果は推論であり、最初はBCE+Diceへの小さな補助項として比較する。[Skeleton Recall Loss, ECCV 2024](https://arxiv.org/abs/2404.03010)、[clDice, CVPR 2021](https://arxiv.org/abs/2003.07311)

複数アノテーターの平均マスクを単純に二値化して骨格化すると、不一致領域の意味が変わる。アノテーターごとに骨格を作り平均する案を別実験にする。骨格のみでは太さや偽陽性を制御できないため、面積を扱う主損失は残す。配布spineを使う案は、その教師利用がコンペ規約上許されるか確認するまで採用しない。

EdgeAttNetはU-Netのボトルネックで入力由来の学習可能エッジをattentionのQ/Kへ組み込み、MAGFiLOでbarbの復元改善を報告する。同じ観測領域という点で優先する専用モデル候補。ただし論文のセグメンテーション性能をそのまま本コンペのインスタンスPQへ換算できない。[EdgeAttNet論文](https://arxiv.org/abs/2509.02964)

### インスタンスモデルを使う条件

Mask2Formerはmasked attentionによるsemantic/instance/panoptic segmentationを扱う。RF-DETRの公式実装にもsegmentationモデルがある。[Mask2Former論文](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html)、[RF-DETR公式実装](https://github.com/roboflow/rf-detr)

高解像度U-Netで画素精度が改善しても分裂・結合が主要誤差として残る場合に着手する。粗い個体マスクを提案し、元解像度クロップ＋提案マスクを入力するU-Netで輪郭を整える案が有力。検出で落とした個体はrefinerで回復できず、長い湾曲個体の箱には多くの背景が入る点を検証する。

インスタンス学習では異なるアノテーターの同一個体を全部別インスタンスとして連結しない。画像ごとに注釈セットをサンプリングするなど、独立した教師集合として扱う必要がある。

## 最小限の比較実験

| 実験 | 変更 | 確認すること |
| --- | --- | --- |
| E0 | main U-Netを同一条件で再評価 | 既存成果物との差、比較基準の確定 |
| E1 | EfficientNet-B3 U-Net、512px | 事前学習モデルへの置換効果 |
| E2 | ConvNeXt-Tiny U-Net、512px | バックボーン差 |
| E3 | E1/E2勝者を768px、元画素間隔を維持 | 文脈拡大による長い個体の改善 |
| E4 | 勝者にSkeleton Recall補助項 | 断片化低減と誤結合増加の差し引き |
| E5 | 境界の失敗が残ればエッジ補助ヘッド／EdgeAttNet | near_missとmatched IoUの改善 |

初回は既存seed=0・物理画像単位15%検証分割に合わせる。同じ画像のアノテーターをtrain/validationに跨がせない。損失・入力・augmentation・後処理探索範囲は揃え、事前学習encoderの学習率だけは小さな候補集合で調整し記録する。epoch数だけでなく学習時間・見たタイル数も記録する。

確率マップは元解像度で評価する。メモリ上全フレーム推論が難しい候補はオーバーラップタイル＋窓ブレンドにし、その推論方式でもE0を評価する。タイル境界での長い個体の分断、端のpadding、元サイズへの復元を確認する。GPUメモリ・所要時間は未計測で、事前に「T4で何時間」と保証しない。

主指標はアノテーター×画像で集計する既存PQ。SQ/RQ、面積・コントラスト別missed、near_miss、fragmented/merged、推論時間、最大GPUメモリを併記する。後処理は固定条件と候補共通の探索条件の両方で比較する。特に最小面積閾値で小個体の改善を消していないか調べる。

同一検証集合で後処理・モデルを選んだPQは選択バイアスがある。候補を絞ってから物理画像単位の複数foldで再確認し、近接観測日をまとめた分割も感度分析として追加する。PQ差の不確実性は物理画像単位のpaired bootstrapで確認する。アノテーター違いの行を独立サンプルとは扱わない。

## 実装する場合の配置

- `models/`：事前学習U-Netの定義・高解像度補助枝。
- `solution/pretrained_unet/`：train/predict、候補設定、追加損失。既存baselineを比較用に維持する。
- `scoring/` と `error_analysis/`：既存の評価を再利用。
- 学習成果物：encoder名、事前学習重みID、入力正規化、分割ファイル一覧、seed、依存バージョン、後処理条件を保存する。

未検証事項は、実GPUでの速度・メモリ、事前学習の転移効果、追加損失のPQ改善、インスタンス提案器の必要性。既存調査にある公開LBスコアや他者の自己申告PQは、今回の順位付けの根拠にしていない。
