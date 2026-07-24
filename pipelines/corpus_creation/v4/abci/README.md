# ABCI 3.0実行補助

`generate_dialogues.pbs`はABCIの`rt_HG`でQwen3-Swallow Student SimulatorをvLLM起動し、教師APIとの候補対話生成を行うPBS例である。

## 準備

```bash
cd v4
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-abci.txt
cp .env.example .env
```

`.env`へ`OPENAI_API_KEY`を設定し、PBSの`CHANGE_TO_YOUR_ABCI_GROUP`を実際のグループ名へ変更する。

`requirements-abci.txt`はvLLMを0.25.1へ固定する。PBSは`config.json`のモデル、Hugging Face revision、vLLM版を読み、版不一致なら生成前に停止する。Qwen3-Swallowは`--reasoning-parser qwen3`で起動する。

## 候補生成

```bash
qsub abci/generate_dialogues.pbs
qstat
```

## Batch工程

候補生成後、次を順に実行する。

```bash
source .venv/bin/activate
python run_v4.py submit-audit
```

初回監査Batch完了後：

```bash
python run_v4.py collect-audit
python run_v4.py submit-repair
```

Repair Batch完了後：

```bash
python run_v4.py collect-repair
python run_v4.py submit-reaudit
```

再監査Batch完了後：

```bash
python run_v4.py collect-reaudit
python run_v4.py finalize
python run_v4.py status
```

Repairは対話全体を参照し、全対象ターンをまとめて修正する。再監査は修正済み対話の全教師ターンを対象にする。対象0件の工程は空結果として記録される。未完了Batchの`collect-*`は状態だけを保存するので、完了後に同じコマンドを安全に再実行できる。

再開時に設定またはpromptが既存manifestと異なる場合は停止する。設定変更時は新しい`output_dir`を使い、通常の再開では`--overwrite`を付けない。
