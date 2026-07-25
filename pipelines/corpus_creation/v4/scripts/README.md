# v4 scripts

主要処理は`../run_v4.py`へ実装済みである。

1. E2/E3事前対応表に基づく候補対話生成
2. 6項目・60点とハード条件によるKeep／Repair／Reject監査
3. Repair対象を対話全体の文脈でまとめて修正
4. 修正済み対話の全教師ターン再監査
5. 対話単位採択とmessages JSONL変換
6. 採択経路、感情、プロフィール、完了状態、支援変更の集計
7. prompt hash、モデルrevision、実行条件による再開時検証

`build_problem_profile_assignments.py`は、`math_train_0`以降の問題を概念規則で分野・必要範囲へ分類し、プロフィールの自然言語`prior_knowledge`と照合してE2/E3プロフィール、初期感情、誤概念モデルを事前対応付けする。行列、ベクトル、二項係数、総和、複素数、床・天井関数、関数合成、二次曲線などを明示的に検出し、モデルAPIは使用しない。

```bash
python scripts/build_problem_profile_assignments.py
python scripts/build_balanced_problem_selections.py
```

1本目は同梱した`questions/translated_1000_math.jsonl`を読み、先頭800件と後半200件で範囲関係の均等化を独立に行う。2本目は先頭800件からコーパス用120件、後半200件からtest-v4用120件を選び、いずれも4種類の`scope_relation`を各30件とする。test-v4では同梱した共通除外18件を選択前に除く。出力先はすべて`assignments/`であり、兄弟フォルダーの`model_evaluation`を必要としない。選択時には`mastered`への未習概念混入、範囲関係不整合、要人手確認問題を機械検査する。選択はseed 42で決定的に再現される。

## OpenAI APIによる学習範囲別パイロット

`pilot_openai_scope_relations.py`は、`corpus_120_selection.json`の均等選択120件から`mastered`、`frontier`、`one_step_beyond`、`far_beyond`を各1件選び、生徒・教師・JudgeをOpenAI APIで実行する。生徒の知識境界、初期感情、事前試行履歴への追従性と、教師の足場かけを小規模に確認するためのスクリプトである。

```bash
python scripts/pilot_openai_scope_relations.py
```

出力先は`data/openai_scope_pilot_v2/results.jsonl`である。既存結果の誤上書きを避けるため、更新時だけ`--overwrite`を指定する。現行結果の分析は`data/openai_scope_pilot_v2/REPORT.md`、設計変更前との比較には`data/openai_scope_pilot/REPORT.md`を参照する。

## SFT系列長監査

`audit_sft_lengths.py`は対象モデルのchat template適用後のtoken数を全件集計し、上限超過があれば終了コード2で停止する。`transformers`が必要であり、`requirements-abci.txt`で構築した環境ではvLLMの依存関係として利用できる。別環境では明示的に`transformers`を導入する。

```bash
python scripts/audit_sft_lengths.py data/run_100_ess_e2e3/v4_sft.jsonl \
  --model TARGET_BASE_MODEL \
  --revision TARGET_REVISION \
  --max-length 4096 \
  --output data/run_100_ess_e2e3/sft_length_audit.json
```

このスクリプトは自動切り詰めや自動分割を行わない。超過時はSFT側の前処理方針を決めてから再監査する。

`audit_sft_lengths.py`は対象base modelのchat templateで全SFTレコードをtokenizeし、上限超過IDをJSONへ保存する。超過があれば終了コード2となる。

```bash
python scripts/audit_sft_lengths.py data/run_100_ess_e2e3/v4_sft.jsonl \
  --model YOUR_BASE_MODEL \
  --revision YOUR_MODEL_REVISION \
  --max-length 4096
```
