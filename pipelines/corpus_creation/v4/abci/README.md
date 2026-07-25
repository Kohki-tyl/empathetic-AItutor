# ABCI 3.0実行補助

`generate_dialogues.pbs`はABCIから教師・生徒のOpenAI APIを呼び、候補対話を生成するPBS例である。`student_provider=vllm`の旧構成も分岐として残している。

## 準備

```bash
cd v4
module load python/3.12/3.12.9
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

`.env`へ`OPENAI_API_KEY`を設定する。`.venv`と`.env`は秘密情報や環境依存ファイルなのでリポジトリへ含めず、ABCIへコピーした`v4/`直下で作成する。

現在の`config.json`は`student_provider=openai`と`student_model=gpt-5.4-mini`を指定する。この場合、PBSはCUDAとvLLMを起動しない。旧vLLM構成を明示的に選んだ場合だけ、CUDA 13.0.1をロードして`requirements-abci.txt`のvLLM 0.25.1との一致を検証する。

`questions/`、コーパス用・test-v4用の選択表、対応表はv4内に同梱している。リポジトリ外の`../questions`や`model_evaluation`は参照しない。投入前検査は次で実行できる。

```bash
python run_v4.py preflight --config config.json
python -m unittest discover -s tests -p 'test_*.py'
```

## 候補生成

```bash
export ABCI_GROUP=YOUR_ABCI_GROUP
qsub -P "${ABCI_GROUP}" abci/generate_dialogues.pbs
qstat
```

ABCIグループはPBSファイルへ埋め込まず、`qsub -P`で指定する。OpenAI生徒ではTCPポートを使わない。旧vLLM構成だけ空きポートを実行時に選ぶ。

## Batch工程

候補生成後、次を順に実行する。

```bash
module load python/3.12/3.12.9
module load cuda/13.0/13.0.1
source .venv/bin/activate
set -a
source .env
set +a
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
