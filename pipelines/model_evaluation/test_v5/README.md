# test-v5: GPT-5.4生徒によるtest-v4対応評価

test-v5は、test-v4の教師、問題60問、類似問題、8プロフィール、6初期感情、scope割当、prompt、状態検証、sampling、seed、Judgeを維持し、生徒モデルだけをOpenAI APIのGPT-5.4へ置換する。

## 固定条件

| 項目 | v5設定 |
| --- | --- |
| 生徒モデル | `gpt-5.4-2026-03-05` |
| API | Chat Completions |
| reasoning effort | `none` |
| temperature | `0.6` |
| top_p | `0.95` |
| top_k | `0`（OpenAI APIに同等パラメータがないため無効化） |
| min_p | `0.0` |
| max completion tokens | `4096` |
| seed | `42` |
| 最大Phase 1ターン | `10` |
| 問題・profile・prompt | `../test_v4`の固定ファイルを直接参照 |
| Judge | test-v4と同じ評価器・prompt |

`gpt-5.4` aliasではなくsnapshotを固定し、manifestと各生成レコードへmodel ID、provider、snapshotを保存する。公式モデルページではGPT-5.4のChat CompletionsとStructured Outputs対応、およびsnapshot `gpt-5.4-2026-03-05`が公開されている。

- [GPT-5.4 model](https://developers.openai.com/api/docs/models/gpt-5.4)
- [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

## v4からの意図的な差分

生徒推論のみがQwen3-Swallow vLLMからGPT-5.4 APIへ変わる。OpenAI Chat CompletionsにはvLLMの`top_k`と`min_p`に相当する入力がないため、`top_k=0`、`min_p=0.0`とする。temperature、top_p、seed、構造化出力schema、最大トークン、状態検証、再試行回数は維持する。

教師はtest-v4と同じvLLM起動・LoRA検証を行う。PBS資源も比較条件を揃えるため`rt_HF`の8 GPUノード占有を維持するが、GPUを使うのは教師のGPU 0だけで、生徒はOpenAI APIを使う。

## ローカル検証

```bash
cd SFT_abci/test/test_v5
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9

/path/to/v4/.venv/bin/python -m unittest test_v5.py
bash -n abci/run_generation.pbs
```

## ABCI一次60問

リポジトリrootの`.env`へ`OPENAI_API_KEY`または`GPT_API_KEY`を設定する。

```bash
cd SFT_abci/test/test_v5

export ABCI_GROUP=YOUR_ABCI_GROUP
export TEACHER_MODEL_PATH=tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5
export TEACHER_REVISION=b1f8317099a97e790ec872c1225ca155979b4816
export TEST_CONDITION=lora
export TEACHER_SERVED_MODEL=v4-sft-provisional
export TEACHER_LORA_PATH=/absolute/path/to/final_adapter

qsub -P "${ABCI_GROUP}" \
  -v TEST_CONDITION,TEACHER_MODEL_PATH,TEACHER_REVISION,TEACHER_SERVED_MODEL,TEACHER_LORA_PATH \
  abci/run_generation.pbs
```

Base教師を使う場合は`TEST_CONDITION=base`として`TEACHER_LORA_PATH`を渡さない。出力は既定で次へ分離する。

```text
data/gpt54_student/<teacher-condition>/primary_60/dialogues.jsonl
```

## 失敗分のみ再試行

初回結果を直接変更せず、別の`retry_60`へ複製してから失敗行だけを再生成する。

```bash
qsub -P "${ABCI_GROUP}" \
  -v TEST_CONDITION,TEACHER_MODEL_PATH,TEACHER_REVISION,TEACHER_SERVED_MODEL,TEACHER_LORA_PATH,\
RETRY_SOURCE_FILE=/absolute/path/primary_60/dialogues.jsonl,\
OUTPUT_FILE=/absolute/path/retry_60/dialogues.jsonl,MAX_GENERATION_PASSES=3 \
  abci/run_generation.pbs
```

## Judge

初回採用分は生成結果と別ファイルへ保存する。

```bash
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
set -a
source ../../../.env
set +a

../../../v4/.venv/bin/python evaluate_in_context_dialogues.py \
  --input data/gpt54_student/lora/primary_60/dialogues.jsonl \
  --output data/gpt54_student/lora/primary_60/evaluated_initial_successes.jsonl \
  --initial-successes-only
```

retry後は、初回Judgeを再利用して新規採用分だけ評価する。

```bash
../../../v4/.venv/bin/python evaluate_in_context_dialogues.py \
  --input data/gpt54_student/lora/retry_60/dialogues.jsonl \
  --output data/gpt54_student/lora/retry_60/evaluated_successes.jsonl \
  --successful-generations-only \
  --reuse-evaluated-from data/gpt54_student/lora/primary_60/evaluated_initial_successes.jsonl
```
