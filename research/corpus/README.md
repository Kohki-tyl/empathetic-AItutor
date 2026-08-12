# 397対話コーパス

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
