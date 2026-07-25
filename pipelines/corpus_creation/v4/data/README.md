# v4 data

再設計後の実行では`run_100_ess_e2e3/`以下へ次を生成する。`data/candidate_dialogues.jsonl`の9件は再設計前pilotであり、新runへ追記しない。

- `candidate_dialogues.jsonl`：正常生成された候補対話
- `generation_errors.jsonl`：生成失敗候補
- `batches/audit_input.jsonl`：初回監査Batch入力
- `turn_audits.jsonl`：教師ターン単位の初回監査
- `batches/repair_input.jsonl`：対話単位の文脈整合Repair入力
- `dialogue_repairs.jsonl`：Repair結果
- `batches/reaudit_input.jsonl`：修正済み対話の全教師ターン再監査入力
- `turn_reaudits.jsonl`：全ターン再監査結果
- `v4_corpus.jsonl`：最終採択コーパス
- `v4_sft.jsonl`：単一messages JSONLのSFTデータ
- `manifest.json`：run fingerprint、設定、prompt hash、Batch ID、採択統計
- `corpus_report.md`：結果概要

学習・検証分割はここでは行わず、SFT実行時に設定する。`manifest.json`は再開時の設定混在を防ぐための実行状態でもあるので、手動で書き換えない。

問題は`math_train_0`から数値順に固定し、候補番号と一対一に対応させる。事前対応表は`../assignments/problem_profile_assignments.jsonl`に置く。候補の`generation_condition`にはE2カリキュラム注釈、E3誤概念、プロフィール、初期感情、初回応答条件を保存するが、SFT messagesへは含めない。
