# v4コーパス作成

v4は、数学的正確性、誤りの診断と回復、認知的共感、感情認識・情緒的支援、適応的足場かけ、理解確認・完了判定を各10点で監査し、高品質な正例を作るSFT用パイプラインである。6項目すべて8点以上かつ全ハード条件を満たす教師ターンだけをKeepとする。

## 採択方針

- 初回監査ですべての教師ターンがKeepなら、その対話を採択する。
- 教師ターンだけを置換して前後の固定文脈と整合させられる場合はRepairとする。
- 同一対話のRepair対象は、対話全体・全監査結果・前後の生徒発話を入力し、まとめて修正する。
- 生徒発話とRepair対象外の教師ターンは変更しない。
- Repair後は修正済み対話の全教師ターンを再監査し、すべてKeepの場合だけ採択する。
- 初回Reject、Repair失敗、再監査で非Keepとなった対話は採択しない。Repairは反復しない。
- 完了・未完了の両方を採択対象とし、train／validation分割はSFT設定側で行う。
- Good／Bad対と指導戦略ラベルは作らない。

厳密な条件は[ADOPTION_CRITERIA.md](./ADOPTION_CRITERIA.md)、モデル仕様は[MODEL_SELECTION.md](./MODEL_SELECTION.md)、SFT形式は[SFT_FORMAT.md](./SFT_FORMAT.md)、設計と妥当性は[V4_CORPUS_DESIGN_AND_FEASIBILITY.md](./V4_CORPUS_DESIGN_AND_FEASIBILITY.md)を参照する。

## 実行環境

教師生成は`gpt-5.6-terra`の通常Chat Completions、初回監査・対話単位Repair・全対話再監査はBatch Chat Completionsを使う。生徒役はABCI上の`tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2`である。

Qwen3-SwallowはthinkingのON/OFF切替をサポートしないため`enable_thinking=False`を指定しない。vLLMへ`--reasoning-parser qwen3`を渡し、Hugging Face revisionとvLLM版をconfigで固定する。

```bash
cd v4
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-abci.txt
cp .env.example .env
```

`.env`へ`OPENAI_API_KEY`を設定する。ABCIでは`abci/generate_dialogues.pbs`がconfigのモデル名・revision・vLLM版を検証してStudent Simulatorを起動する。

手動起動する場合は次の設定を使う。

```bash
vllm serve tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2 \
  --revision 496cd5558fef4af1d426e96327d7a74681063280 \
  --served-model-name tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2 \
  --reasoning-parser qwen3 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --port 8001
```

## 実行順序

```bash
python run_v4.py generate
python run_v4.py submit-audit
python run_v4.py collect-audit
python run_v4.py submit-repair
python run_v4.py collect-repair
python run_v4.py submit-reaudit
python run_v4.py collect-reaudit
python run_v4.py finalize
python run_v4.py status
```

対象0件のRepair／再監査は`skipped`として安全に処理される。Batchが未完了なら、完了後に同じ`collect-*`を再実行する。通常の再開時に`--overwrite`は付けない。

## 再現性と再開

manifestへ、変更不可設定、モデル、reasoning effort、seed、student revision、vLLM版、プロンプトSHA-256、実行環境、Batch ID、run fingerprintを保存する。再開時にfingerprintが変わっていれば処理を停止する。

`target_dialogues`と`max_candidates`だけは追加生成のため変更でき、履歴をmanifestへ残す。プロフィール×初期感情の割当は24件単位の決定的ブロックで作るため、候補数を増やしても既存candidateの割当は変わらない。最初から別条件で実行する場合は別の`output_dir`を使うか、意図を確認したうえで`generate --overwrite`を使う。

## 生徒条件

V2-S01〜V2-S04と6初期感情の24組合せを層化する。初期感情はプロフィールから独立させ、最初の生徒発話では維持する。理解度は1ターン最大1、確信度は最大0.25、感情は定義済みサイクルの隣接状態だけ変化できる。獲得知識の巻き戻し、内部状態の発話への露出、教師役への逸脱を禁止する。

問題はseed付きでシャッフルし、プロフィールとは独立に割り当てる。問題の学習範囲・難度とプロフィールの自動対応付けはv4では実装しない。範囲不一致は採択後の分析項目かつ研究上の制約として扱う。

## 出力

`data/run_100/`へ候補対話、生成エラー、Batch入力、初回監査、文脈付きRepair、全対話再監査、採択コーパス、SFT JSONL、manifest、レポートを保存する。詳細は[data/README.md](./data/README.md)を参照する。

SFT出力は問題と最初の生徒発話を同一userに入れ、roleを交互にする。assistantは短い監査可能な`<analysis>`と生徒向け`<final>`を学習対象にする。使用モデルのtokenizerで全件の系列長を確認し、assistant targetを途中で切り詰めない。
