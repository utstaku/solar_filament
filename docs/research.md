# コンペ仕様の要点とアーキテクチャ調査（2026-09-19時点）

凡例: ✅ = 一次情報で確認済み（Kaggle API / 主催者ノートブック / 論文） / 💭 = 推論・未検証の見立て

## 1. コンペ仕様 ✅

- **タスク**: フィラメントを個体ごとにマスク化（実質インスタンスセグメンテーション）。キラリティ・spine・bboxは採点対象外。
- **指標**: 2026-08-07に mean Dice → **Panoptic Quality** へ変更。
  `PQ = Σ_TP IoU / (|TP| + 0.5|FP| + 0.5|FN|)`、TPは IoU > 0.5 のペア。
  主催者実装では**アノテーター×画像単位**で集計し、同じ予測を各アノテーターの注釈セットそれぞれと照合する。
- **提出**: CSV（`filament_id` = `<画像ID>_<連番>`, `segmentation_rle` = pycocotools圧縮RLEのcounts、サイズ2048×2048固定）。
  コードコンペではない。1日5回、最終選択2つ。Public LBはtest 180枚の約50%。
- **ルール**: 外部データ可。ただし推論に使えるのは配布test画像のみ、学習に「他のground-truthメタデータ」を使うのは不可。
- **最終評価**: 定量70%（PQ、Dice/IoU分布、one-to-many / many-to-one の分布）＋定性30%（パイプライン説明、形態の見た目、**コード品質**）。
  計算効率も考慮。公開Gitリポジトリ・requirements.txt・パイプラインnotebook・4ページレポートが必須。
- 主催者コメント（8/20）: 「PQ 0.35超は我々にとって大きな価値がある」。

## 2. データ ✅

- train 707枚 / test 180枚、2048×2048 8bitグレースケールJPEG（FITSから変換）、GONG 6サイトがほぼ均等、2011–2022年（2011–2017が大半）。trainとtestの年・サイト分布はほぼ同じ。
- アノテーション: COCO形式、フィラメント8,199個。1注釈セットあたり 最小1 / 中央値7 / 最大26個。
  polygon（`segmentation`）、`bbox`、`spine`（中心線ポリライン）、キラリティ（Left 2,535 / Right 2,590 / Unidentifiable 3,074）。
- **個体の形状**: 全8,199個体が**単一polygon**（複数polygonに分かれた個体は0件）＝GTの1個体は必ず1つの連結領域。面積は中央値1,228px、5%点317px、99%点13.7kpx。前景は全画素の0.36%しかない。目視では幅10〜20px程度、周囲より10〜15%暗いだけの低コントラストで、barbまで細かく縁取られている。
- **画像の癖**: 円盤外は真っ黒ではなく灰色（輝度30〜40）で、縁に明るいリングがある。円盤検出は固定閾値ではなく大津＋面積等価半径が安定（`preprocess/disk.py`、`solution/classical_baseline/` で確認）。
- **複数アノテーター**: `images` は1,154エントリだが物理画像は707枚。同一画像に最大3セットの独立注釈（ID先頭の `050101-` / `050102-` … が違う）。MAGFiLO論文のアノテーター間一致度は κ = 0.66 → **PQの実質的な上限は1よりかなり低い** 💭。
- **時間的近接**: test画像の 36% は24時間以内、77% は72時間以内に撮影されたtrain画像がある（ファイル名の時刻から算出）。フィラメントは数日〜数週間持続するので、testとほぼ同じ構造がtrainに写っている。
  → LBに近いCVは「物理画像（file_name）でグループ化したランダムK-fold」。日付でグループ化すると悲観的な見積もりになる 💭。どちらでも同一file_nameのアノテーター違いを跨がせないことは必須。

## 3. リーダーボードの状況と注意 ✅

- 首位0.62、0.55に大集団。人気の公開ノートブック「Unet Segmentation | 0.55+」は、学習したモデルの予測ではなく
  **lzma+base85で埋め込まれた事前計算済み提出CSV**（`CHAMPION_PAYLOAD`）を出力している。出所は不明。
- 完全版MAGFiLO（1,593観測 / 10,244フィラメント）は公開データで、test画像の正解を含みうる。別の公開ノートブックも
  「test注釈を含むMAGFiLOミラーは使っていない」と明記している。これは上記ルールに抵触するので**使わない**。
- 正攻法の公開パイプライン（YOLO検出→クロップU-Net）の自己申告ローカルPQは約0.41。当面の現実的な目標は OOF PQ 0.40〜0.45 💭。
- 自前の古典的ベースライン（背景正規化→閾値→closing→連結成分、`solution/classical_baseline/`）は train 60枚で **PQ ≈ 0.11**（TP 175 / FP 1,128 / FN 547）✅。低コントラストと彩層のテクスチャで偽陽性が多発し、閾値処理では歯が立たない。これが下限の目安。
- 最終順位はコード公開＋再現性検証＋定性評価込みなので、LBの数字より「再現可能で説明できるパイプライン」が重要。

## 4. 先行研究

| 研究 | 手法 | データ | 備考 |
| --- | --- | --- | --- |
| Ahmadzadeh+ 2024 (Sci. Data) | MAGFiLOデータセット | GONG Hα | 本コンペのデータ源 |
| EdgeAttNet (arXiv 2509.02964) | U-Net + ボトルネックの自己注意のQ/Kを学習可能なエッジマップで変調 | MAGFiLO | barb（棘状構造）の再現が改善、軽量 |
| MORDEN (arXiv 2607.24525) | 5×5カーネルU-Net + IN + 階層的ピラミッドプーリング + パッチ注意、focal loss | CHASE/HAS等 | IoU 0.70–0.81。後処理にDenseCRF（境界）と**DBSCAN（断片の統合）** |
| Flat U-Net (ApJ 2025) | 超軽量U-Net | HSOS全球Hα | 効率重視 |
| Diercke+ 2024 | YOLOv5検出 + U-Net | ChroTel | 検出→セグメンテーションの2段 |
| Reche & Cid 2024 | DETR検出 + U-Netインスタンス + 追跡 | — | 2段構成 |

既存研究はほぼ「U-Net系セマンティック＋後処理」。インスタンス単位のPQで評価した先行研究は見当たらず、**断片の統合／過剰マージの分離**が未解決の中心課題（主催者も Description で明言）。

## 5. 有望なアーキテクチャ

指標の性質から逆算すると、必要なのは (a) 細い構造でIoU > 0.5を超えるための**ネイティブ解像度でのピクセル精度**、(b) **個体の正しいグルーピング**、(c) 小さな偽陽性・見逃しの抑制、の3つ。

### A. 検出 → クロップ → リファイン（2段構成）— まず作るべき本命 💭
- 検出器: YOLO11/YOLO26（-seg含む）を imgsz 1280–2048、または RF-DETR(-Seg)。リファイナ: `segmentation_models_pytorch` の U-Net / UNet++（EfficientNet-B3〜B4、ConvNeXt-Tiny）に `[画像, CLAHE, シード(箱 or 粗マスク)]` の3chを入力、クロップはネイティブ解像度。
- 長所: 個体分離を検出器に任せられる。リファイナは小クロップなのでT4で十分速い。公開ノートブック群で実績（ローカルPQ≈0.41）。Kaggle Sartorius細胞コンペ1位と同じ構図。
- 短所: 細長く湾曲した物体は箱の中が背景と隣の個体だらけになり、NMSで隣接フィラメントが消えやすい。箱のジッタに対するリファイナの頑健性（学習時にGT箱を揺らす）が要。

### B. フル解像度セマンティック + インスタンス化後処理 — ピクセル精度の土台 💭
- U-Net / UNet++ / UPerNet + ConvNeXt・EfficientNetV2・MaxViT エンコーダ（timm / smp）。**縮小せず** 768–1024pxタイルで学習、推論はオーバーラップ＋コサイン窓ブレンド、D4 TTA。
- 損失: BCE + Dice（またはTversky, β>α で再現率寄り）に **clDice** か **Skeleton Recall Loss**（ECCV 2024、CPUで骨格を前計算するので軽い）を加えて連結性を保つ。
- **補助ヘッド**: 配布JSONの `spine`（中心線）を教師にした中心線ヒートマップ＋境界マップ。spineは他コンペにない強い教師信号で、断片の統合（端点距離＋方向の整合で接続）や、接触個体の分離（spineをシードにしたwatershed / 埋め込みクラスタリング）に使える。
- 複数アノテーター: 同一画像の注釈を平均した**ソフトラベル**で学習（Kaggle Contrails 1位の要点）。
- 短所: PQを落とすのは結局グルーピングのヒューリスティック部分。連結成分だけでは断片化する。

### C. マスク分類型のend-to-endインスタンスセグメンテーション — グルーピング問題の切り札候補 💭
- Mask2Former / MaskDINO（Swin・ConvNeXt・DINOv3バックボーン）、RF-DETR-Seg（DINOv2バックボーン、ICLR 2026、小規模データのfine-tune向け設計）。ベースラインとして Mask R-CNN / YOLO-seg（公開ノートブックに yolov8l-seg imgsz=2048 の例）。
- 長所: 個体のグルーピングをヒューリスティックではなく学習で解ける。箱NMSが不要（Mask2Former系）なので、細長い個体が隣接・交差していても消えにくい。
  なお GT の個体は必ず単一の連結領域（§2）なので、「非連結な断片を1個体にまとめる」能力そのものは不要。必要なのは低コントラストで途切れて見える部分を**連結したまま**出力すること。
- 短所: マスク出力が入力の1/4解像度 → 幅数pxの構造でIoUが落ちる。707枚ではデータ不足気味。2048入力はT4 16GBで厳しい（タイル化すると長いフィラメントが切れる）。
- → 単体よりも「個体の提案器」として使い、ピクセルはBで決める **D** の形が有力。

### D. ハイブリッド（到達目標）💭
B のフル解像度セマンティックマスクの各ピクセル／連結成分を、A または C が出す個体提案に割り当てる（panoptic fusion）。「どこがフィラメントか」はB、「どの個体か」はA/C。重なりは確信度順に排他化。閾値・最小面積・統合距離はすべてOOFのPQで調整。

### E. 基盤モデル 💭
- SAM 2 / SAM 3 を箱プロンプト付きリファイナとして fine-tune: 境界の事前知識は強いが、自然画像由来で「暗く細い低コントラスト構造」への転移は不確実。重く、計算効率の採点でも不利。優先度は中〜低、A のリファイナとの差し替え実験として。
- 太陽物理の基盤モデル（NASA-IBM Surya、SDO-FM）: SDO/AIA・HMIの多波長4096²が前提で、地上Hα単チャネルとは入力が別物。優先度低。
- DINOv3バックボーン（ConvNeXt/ViT、衛星画像事前学習版あり）: 密な特徴が強くB/Cのエンコーダ候補。ただしライセンスが独自なので、公開義務との整合を先に確認。

### F. 細線構造の専用アーキテクチャ 💭
DSCNet（Dynamic Snake Convolution, ICCV 2023）、CS-Net、EdgeAttNet。2048²では重い／実装コストの割に、損失（clDice / Skeleton Recall）と補助ヘッドで同等の効果を狙う方が安い。EdgeAttNetのエッジ誘導注意はボトルネックのみなので B への追加実験としては手頃。

## 6. 実務上の推奨 💭

- **前処理**: 円盤検出 → 周縁減光の除去（動径方向のメディアンで正規化）→ 円盤外マスク。入力は `[原画像, 平坦化画像, CLAHE]` の3chにしてImageNet事前学習エンコーダを流用。
- **Augmentation**: 採点対象はマスクだけなので flip・90°回転・円盤中心まわりの任意回転はすべて有効。シーイング劣化（blur）、ノイズ、ガンマ、JPEG圧縮。
- **CV**: file_name でグループ化した5-fold、サイト×年で層化。モデル選択は必ず主催者実装と同じPQ（SQ/RQ分解も見る）で。
- **後処理の調整対象**: 確率閾値、最小面積、穴埋め、断片統合（端点距離＋spine方向）、個体間の重なり除去。
- **計算資源**: Kaggle GPU（T4×2 / P100、週30h）。707枚なのでBの1fold学習は1〜2時間程度の見込み。

## 7. 進め方の案

1. （済）古典的ベースラインでパイプライン疎通と参考PQ（≈0.11）→ `solution/classical_baseline/`
2. EDA kernel: フィラメント幅の分布（面積/spine長）、サイト別の画質差、**アノテーター同士のPQ（＝人間レベルの上限）**
3. B のベースライン（smp U-Net + 連結成分）と A のベースライン（YOLO-seg 2048）を同じ5-foldのOOF PQで比較
4. 強い方を軸に: ソフトラベル、clDice/Skeleton Recall、spine補助ヘッド、断片統合、TTA、foldアンサンブル
5. グルーピングがボトルネックのままなら C（RF-DETR-Seg / Mask2Former）を個体提案器として追加し D へ

## 出典

- コンペ各ページ（Kaggle API `competitions pages` で取得）、主催者の [Self-Evaluation Notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
- [MAGFiLO (Scientific Data 2024)](https://www.nature.com/articles/s41597-024-03876-y) / [IEEE BigData Cup 2026 ページ](https://bigdataieee.org/BigData2026/cup/solar-filament-segmentation/)
- [EdgeAttNet](https://arxiv.org/abs/2509.02964) / [MORDEN: A Modern ConvNet for Solar Filament Detection](https://arxiv.org/html/2607.24525v1) / [Flat U-Net](https://arxiv.org/abs/2502.07259)
- 公開ノートブック: [YOLO11s Detect → Crop U-Net](https://www.kaggle.com/code/timgluhkikh/best-score-magfilo-yolo11s-detect-crop-u-net)、[New Metric | Solar Filament](https://www.kaggle.com/code/ektarr/new-metric-solar-filament)、[MAGFiLO V9](https://www.kaggle.com/code/muhammadshoaibaltaf/magfilo-v9-evaluation-aligned-solar-filament-s)、[Grouped CV writeup](https://www.kaggle.com/code/dattadhebe/solar-filament-grouped-cv-and-strategy-writeup)
- 参加者の公開リポジトリ: [1wraith/AI-solar-filament-segmentation](https://github.com/1wraith/AI-solar-filament-segmentation)（ConvNeXt U-Net + spine/埋め込みヘッド + clDice）、[mtalafha90/Solar_Filament_Segmentation_Challenge_2026](https://github.com/mtalafha90/Solar_Filament_Segmentation_Challenge_2026)（エッジ誘導注意U-Net、周縁減光除去、方向整合による断片統合）
- [Kaggle Contrails 1位解法（ソフトラベル）](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/writeups/jun-koda-1st-place-solution) / [Skeleton Recall Loss (ECCV 2024)](https://arxiv.org/abs/2404.03010) / [RF-DETR](https://github.com/roboflow/rf-detr)
