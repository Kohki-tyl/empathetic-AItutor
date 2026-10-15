# 研究用コーパス

## 800対話の続き（追加500対話）

[empathetic_dialogues_1300/](empathetic_dialogues_1300/README.md) に、新しく翻訳した500問から生成した対話801〜1300と、既存800対話との結合版を保存している。教師GPT-5.4・生徒GPT-5.4-miniで生成し、追加分は484対話が教師判定で完了、16対話が最大15往復で未完了。追加500対話の内容監査を実施済み。詳細は[監査報告](empathetic_dialogues_1300/audit/README.md)を参照。

## MATH追加500問の基本翻訳

[math_additional_500_gpt6_luna/](math_additional_500_gpt6_luna/README.md) に、既存の `translated_1000_math` と重複しない追加500問を GPT-6 Luna で翻訳した問題・解答ペアを保存している。既存1,000問との結合版も収録している。教師・生徒対話のSFTデータとは異なる。

## 日本語比較学習用コーパス

[baselines_ja1000/](baselines_ja1000/README.md) に、Math-SFT 用の [MathDial 日本語版](baselines_ja1000/math_ja_1000.jsonl) と Empathy-SFT 用の [EmpatheticDialogues 日本語版](baselines_ja1000/empathy_ja_1000.jsonl) を各1,000対話保存している。GPT-6 Luna による機械翻訳で、原文・抽出条件・検査報告・対訳資料も同梱する。1,000件は学習・検証の合計であり、意味の忠実性の人手レビューは未実施。

## 397対話コーパス

本ディレクトリには、教師の全発話について監査を通過し、対話が完了している397件の対話コーパスを収録する。

## 収録ファイル

- `v3_397_dialogues.jsonl`: 評価・学習で利用する対話データ本体
- `v3_397_metadata.jsonl`: 各対話のsource ID、参照解答、難度、旧プロフィールIDなどの評価用メタデータ
- `v3_397_dialogues_manifest.json`: 件数、抽出条件、出典、ハッシュ値

## データ形式

1行が1対話のJSONLであり、各行はOpenAI chat形式の `messages` 配列を持つ。各対話にはsystem発話が1件あり、その後にuser発話とassistant発話が並ぶ。assistant発話は `<analysis>...</analysis><final>...</final>` 形式である。

## 出典と選定条件

- 原本: `pipelines/sft/v3/data/v3_cot_sft.jsonl`
- 原本マニフェスト: `pipelines/sft/v3/data/v3_cot_sft_manifest.json`
- 抽出元: `pipelines/corpus_creation/v3/data/v3_rebuilt_corpus.jsonl`
- 抽出条件: `is_completed = true` かつ `all_teacher_turns_audited = true`
- シャッフルシード: `42`

同じ397件から変換されたv3.1版ではなく、監査済みv3 SFTコーパスをそのまま複製している。

## 完全性

- 対話数: 397
- system発話数: 397
- user発話数: 1,626
- assistant発話数: 1,626
- SHA-256: `88a102ddbca4c4d78b669a2646e00f2d32f1d64ac182b322e9f01cc44be40eea`
- 評価用メタデータSHA-256: `13a0b7eeb3bb7bc421573bdbdda31accb349722b61130af6bdeea6060630e1e3`

## 各1,000対話の比較SFTフォルダ

[sft_1000/](sft_1000/README.md) にMath-SFT・Empathy-SFT・EduEmpathy-SFTを各学習900件／検証100件で用意した。生成対話はユーザー指定により表記の品質条件を緩和し、正規化の変更履歴を同梱した。
