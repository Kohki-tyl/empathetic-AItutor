# v4 data

生成物は実行条件ごとのサブフォルダーへ保存し、別runへ追記しない。主要フォルダーの位置付けは次のとおりである。

- `openai_scope_pilot/`：試行履歴制約追加前の4条件pilot
- `openai_scope_pilot_v2/`：教師prompt改善前の4条件pilot
- `openai_scope_pilot_v3/`：改善後教師promptの4条件pilot
- `run_100_ess_e2e3/`：旧vLLM生徒構成の生成試行。最終コーパスには使用しない
- `run_120_slices/`：OpenAI生徒で120候補を分割生成した構造化ログ。標準出力・標準エラーログは一時物としてGit管理しない
- `run_10_openai_gpt54mini/`：履歴上のフォルダー名。120候補の統合監査、Repair、表面修正、最終104件のSFT入力を収録する現在の正本

各runでは必要に応じて次の名前を使用する。

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

最終SFT入力は`run_10_openai_gpt54mini/v4_all_keep_sft_eligible.jsonl`であり、SFT実行フォルダーの`pipelines/sft/v4/data/v4_sft.jsonl`は同一hashの実行用スナップショットである。

問題は`math_train_0`から数値順に並べ、先頭800件だけをコーパス候補プールとする。`../assignments/corpus_120_selection.json`で4種類の範囲関係を各30件、計120件に固定し、後半200件のテスト用問題が混入した場合は生成を停止する。事前対応表は`../assignments/problem_profile_assignments.jsonl`に置く。候補の`generation_condition`にはE2カリキュラム注釈、概念単位の知識境界監査、E3誤概念、プロフィール、初期感情、初回応答条件を保存する。SFT messagesには完全なgeneration conditionを含めず、教師が実行時にも受け取る初期感情ラベルだけを最初のuserへ含める。
