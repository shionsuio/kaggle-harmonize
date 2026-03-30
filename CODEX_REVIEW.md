# Codex用レビュー依頼

## 現状
- Kaggle SDRFメタデータ抽出コンペ（テスト15論文）
- Public LBスコア: 0.315（目標: 0.46+）
- 残り提出回数: 9回

## パイプライン概要
```
PRIDE API anchor → LLM抽出(v2) → build_submission_v2.py → test_overrides.json上書き → submission.csv
```

## レビューしてほしいファイル
1. `pipeline/build_submission_v2.py` — メインのsubmission生成ロジック
2. `pipeline/extract_with_llm_v2.py` — LLM抽出（逐次累積型4ステップ）
3. `pipeline/test_overrides.json` — テスト15本の手動確定値
4. `pipeline/score_function.py` — 評価関数（コンペ提供）

## 分析してほしいこと

### 1. build_submission_v2.py の値選択ロジック
- Priority 1-6の優先度は正しいか
- confirmed vs hypothesis の選択で漏れはないか
- clean_value で正しい値まで消してないか
- fuzzy_snap のcutoff設定は妥当か（カラムごとに変えてる）

### 2. LLM抽出の品質問題
- LLM正解率44%（54本CV）。残り56%の内訳:
  - 34% フォーマット/値不一致
  - 22% 見逃し（NAを返す）
- v3（候補リストから選ばせる方式）は最頻値に引きずられて失敗した
- 改善案はあるか

### 3. 評価関数の特性を活かせてるか
```python
# 評価関数の特徴:
# 1. NT=値のみで比較（AC, TA, MT, PPは無視）
# 2. 80%文字列類似度でクラスタリング
# 3. 正解側カテゴリのみ評価（埋めすぎペナルティなし）
# 4. Modification列は.サフィックス除去して統合
# 5. Not Applicable のみのカラムはスキップ
```
- この特性を最大限活かすための戦略は何か
- 類似度80%のボーダーラインで落ちてる値がないか

### 4. Private LB推定の検証
以下のPXDをPrivateと推定している。妥当か？
- PXD062469（確定）、PXD064564（高）、PXD062877（高）、PXD061285（高）、PXD061195（候補）

### 5. rawファイル構造推論
テスト15本のrawファイル名から実験構造を因数分解した。
- PXD040582: 完全直積 2×4×3=24 → per-file割当実装済み
- PXD016436: 6温度×3rep=18 → 実装済み
- 他にper-file割当できるPXDはあるか

## CV結果（54本訓練データ）
- Average F1: 0.557
- 最もロスが大きいカラム:
  - Label: 0.63 (F1=0が20件)
  - Modification: 0.61
  - FractionIdentifier: 0.36 (F1=0が18件)
  - Disease: 0.23
  - MaterialType: 0.24
  - OrganismPart: 0.34

## 重要な教訓（ここまでで学んだこと）
- 強制NA化は逆効果（0.25→0.16に下がった）
- v3候補リスト方式は失敗（最頻値DTT/IAA/normalに引きずられる）
- CellLineメタデータ（Sex/Age/DevStage）は有効
- Organism括弧除去は必須（類似度0.688<0.80問題）
- 訓練NA率でテスト判断は危険
