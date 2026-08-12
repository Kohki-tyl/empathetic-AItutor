# 確定研究用・教師モデル評価パイプライン

このディレクトリは、[`EVALUATION_FRAMEWORK.md`](../EVALUATION_FRAMEWORK.md) に基づき、数学教師モデルを固定条件で比較するための実行パイプラインである。

現在のテスト条件をまとめた独立した設計書は [`TEST_DESIGN.md`](TEST_DESIGN.md) を参照する。

## 評価方針

- 評価対象は、生徒へ実際に提示された教師発話だけとする。
- 評価単位は対話全体とし、ターン単位の得点は付けない。
- 教師の内部推論は生徒・Judgeへ渡さず、生成レコードにも保存しない。
- 生徒LLMの自然さ、知識境界、状態更新、プロファイル整合性は評価しない。
- 生徒の知識境界違反や数学的誤答を理由に再生成・除外しない。
- 初期応答をケースごとに一度だけ生成し、すべての教師条件で共有する。
- 類似問題正答率は近接転移の副評価とする。

## 同梱する固定評価資産

`assets/` に次の固定資産を同梱する。既定設定とABCI実行は、このディレクトリ外のリポジトリ資産を参照しない。

- 学年、話し方、ケース別の初期状態を持つ8種類の簡易生徒プロファイル
- 翻訳済み評価候補問題200問と参照解答
- 対応する類似問題200問と参照解答
- 問題―プロファイル・初期状態割当200件
- 品質検査による除外18件
- 固定評価100問の学習コーパス `source_id` 漏洩監査
- 学習状況（`mastered`、`frontier`、`one_step_beyond`、`far_beyond`）
- 初期感情
- 翻訳済み問題の末尾200問から選定した評価100問

既存test-v4/test-v5の生徒知識境界検証と生徒らしさJudgeは再利用しない。

`vendor_assets.py` はリポジトリ内の原本から同梱資産を更新する保守用スクリプトであり、ABCI実行時には使用しない。

## 処理フロー

```text
prepare_cases.py
  -> data/cases.jsonl

generate_initial_student_utterances.py
  -> data/initial_student_responses.jsonl

generate_dialogues.py --condition <condition>
  -> data/runs/<condition>/dialogues.jsonl

evaluate_dialogues.py
  -> data/runs/<condition>/evaluated.jsonl

summarize_results.py
  -> data/summary.json + data/summary.md
```

各出力には、同名の `.manifest.json` を作成する。入力ファイル、プロンプト、モデル設定のfingerprintが変わった状態では既存出力を再開しない。

## セットアップ

```bash
cd research/model_evaluation
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PowerShellでは次を使用する。

```powershell
cd research/model_evaluation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`config.example.json` を研究実行用に複製し、生成設定を凍結する。入力パスはすべて `model_evaluation` 内で完結している。Judgeモデルは意図的に空欄であり、本番実行時に明示する。

## 1. 評価ケースの準備

```bash
python build_selection.py
python prepare_cases.py --config config.example.json
```

`build_selection.py` は同梱した翻訳済み問題200問を母集団とし、品質除外18問を維持したうえで、4区分を各25問ずつ固定seedで選ぶ。4区分はそのまま学習状況として生徒LLMへ渡す。各ケースでは、学年、カテゴリ化した話し方、学習状況・初期感情・理由からなる初期状態を1つの生徒プロファイルへまとめ、類似問題と結合する。話し方はタメ口／丁寧口調、自信がある／慎重／控えめ、短い／標準で指定し、数学能力とは結びつけない。

学習コーパス本体は容量が大きいため同梱しない。代わりに、固定100問について実施済みの `source_id` 漏洩監査、監査対象コーパスのパスとハッシュを `assets/training_leakage_audit.json` に同梱する。学習コーパスをABCIにも配置する場合は、`paths.training_corpora` へ指定すると実行時に再検査できる。

## 2. 初期応答の固定

```bash
export OPENAI_API_KEY=...
python generate_initial_student_utterances.py --config config.example.json
```

生徒モデルは `gpt-5.4-2026-03-05` に固定している。初期応答は、生徒プロファイル内の学年、話し方、4区分の学習状況、初期感情と理由から問題ごとに一度だけ生成する。生成後は初期応答もプロファイルの初期状態へ加え、以後の生徒LLM入力で固定する。成功済みケースは再実行しても変更しない。APIエラー、空応答、形式破損だけを固定回数まで再試行する。

生徒プロンプトは過去のtest-v4/test-v5を参考に、感情をラベルではなく語調で示すこと、教師へ無条件に同意しないこと、一度に一段階だけ応答すること、教師口調を避けることを継承する。一方、旧プロンプトの知識境界強制、誤概念の強制維持、作為的な誤答、数値的な内部状態更新は使用しない。

## 3. 教師との対話生成

教師はOpenAI互換エンドポイントで提供する。

教師プロンプトは、過去テストの数学的検算、具体的な感情受容、最小単位の足場、停滞時の支援変更、根拠確認後の完了判定を継承する。内部分析は任意であり、評価対象と生成記録に残すのは `<final>` の可視発話だけである。

```bash
export TEACHER_API_KEY=EMPTY
python generate_dialogues.py \
  --config config.example.json \
  --condition base \
  --teacher-model teacher-base \
  --teacher-base-url http://127.0.0.1:8000/v1 \
  --output data/runs/base/dialogues.jsonl
```

SFT条件は出力と条件名を分ける。

```bash
python generate_dialogues.py \
  --config config.example.json \
  --condition sft-v1 \
  --teacher-model teacher-sft-v1 \
  --teacher-base-url http://127.0.0.1:8000/v1 \
  --output data/runs/sft-v1/dialogues.jsonl
```

初期応答のハッシュは各レコードへ保存される。同一ケースでハッシュが異なる条件は、対応比較時にエラーとなる。

対話生成成功は、API・形式上の失敗なく教師完了または最大ターンへ到達したことを表す。数学的・教育的成功は含めない。類似問題生成の失敗は、副評価の欠測として別に記録し、対話生成成功率を変更しない。

## 4. 対話全体の6軸評価

### 397対話コーパスの固定50件を評価する場合

397対話はシード42で既にシャッフルされているため、その先頭50件を非復元の固定サブセットとして使用する。評価入力は同梱済みだが、次のコマンドで再生成できる。

```bash
python prepare_corpus_evaluation.py
```

生成物は `selections/corpus_v3_50_dialogues.jsonl`、選定記録は `selections/corpus_v3_50.json` である。変換時に教師の内部 `<analysis>` とタグを除去し、初回生徒発話に付加された問題文も分離する。Judgeには通常の評価と同様、問題、参照解答、可視対話だけを渡す。

```bash
python evaluate_dialogues.py \
  --config config.corpus-gpt56terra.json \
  --input selections/corpus_v3_50_dialogues.jsonl \
  --output data/runs/corpus-v3-50/evaluated.jsonl
```

この専用設定はJudgeを `gpt-5.6-terra`、reasoning effortを `high`、temperatureを同モデルが対応する既定値 `1` に固定し、環境変数 `GPT_API_KEY` を参照する。`.env` を自動読込するランチャーを使わない場合は、実行前に同名の環境変数へ読み込む。

実施済みの固定50件評価は `results/corpus_v3_50/` に保存している。`findings.md` が主要結果、`summary.json` と `summary.md` が機械集計、`evaluated.jsonl` がケース単位のJudge出力である。

```bash
export JUDGE_API_KEY=...
python evaluate_dialogues.py \
  --config config.example.json \
  --input data/runs/base/dialogues.jsonl \
  --output data/runs/base/evaluated.jsonl \
  --judge-model YOUR_JUDGE_MODEL
```

Judge入力は、問題、参照解答、可視対話だけである。生徒プロファイル、初期状態、教師内部推論、停止理由、完了フラグ、API call metadataは渡さない。指導完了判定も可視発話の内容だけから行う。

評価軸はすべて0〜10点または `NA` である。

- 数学的正確性
- 誤り診断と回復
- 指導完了判定
- 足場かけ
- 情緒的支援
- 感情把握

`NA` は平均から除外する。対話の全体得点は、適用可能軸の平均を6倍した60点換算値である。重大失敗は得点を上書きせず、発話、種類、影響、回復状況、関連軸を個別に保存する。

## 5. 集計と対応比較

```bash
python summarize_results.py \
  --input data/runs/base/evaluated.jsonl \
  --input data/runs/sft-v1/evaluated.jsonl \
  --output-json data/summary.json \
  --output-markdown data/summary.md
```

主要結果は対話生成成功率と全体得点である。6軸、指導・共感の群別平均、重大失敗率、指導完了率、最大ターン到達率、類似問題正答率を内訳として出力する。教師条件間の全体得点・軸別得点は、共通評価可能ケースによる対応ブートストラップ95%区間を出力する。

## shard実行

`generate_dialogues.py` は `--num-shards` と `--shard-index` に対応する。shardごとに異なる出力を指定し、終了後にJSONLを連結する。異なるmanifestの出力を無検証で上書きしない。

## ABCI

`model_evaluation` ディレクトリだけをABCIへコピーすれば実行できる。

```bash
# ローカル側の例
rsync -av \
  --exclude .venv --exclude __pycache__ --exclude 'data/runs' \
  research/model_evaluation/ USER@ABCI:~/model_evaluation/
```

ABCI上で環境を作成する。

```bash
cd ~/model_evaluation
bash abci/setup_environment.sh
cp .env.example .env
# .envへOPENAI_API_KEYとTEACHER_MODEL_PATHを設定する
```

プロジェクトコードを指定してジョブを投入する。

```bash
qsub -P PROJECT_ID abci/run_generation.pbs
```

`abci/run_generation.pbs` は、同梱資産からの100問選定、ケース準備、固定初期応答生成、教師vLLM起動、条件別対話生成を順に実行する。仮想環境、`.env`、設定、出力はすべてコピー先の `model_evaluation` 内を既定値とする。別の場所を使う場合は `VENV_PATH`、`ENV_FILE`、`CONFIG_FILE`、`OUTPUT_FILE` で上書きできる。

## ローカル検証

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
python build_selection.py
python prepare_cases.py --config config.example.json
```

単体テストでは、末尾200問からの均衡100ケース選定、8種類の簡易プロファイル、4区分の学習状況、プロファイル内の固定初期応答、内部推論の除外、`NA`集計、生成成功率の分母、教師条件間の初期応答一致を検査する。
