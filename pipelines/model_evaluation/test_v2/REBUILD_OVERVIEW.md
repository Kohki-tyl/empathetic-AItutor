# リビルド版 v2-test 概要

## 目的

リビルド版v2-testは、Base Swallow 8Bとv2 CoT-SFT Swallow 8Bの数学指導性能を、
同一の問題・生徒モデル・生徒プロフィール・乱数seedで比較するテストである。

旧テストで確認された以下の問題を解消することを主目的とする。

- 生徒発話へのJSON、内部状態、教師CoTの露出
- 生徒状態更新の解析失敗と未検証状態の採用
- Base教師の未閉鎖`<final>`によるCoT露出
- Judge API接続失敗と、失敗結果の完了扱い
- 実際にロードしたcheckpoint・adapterの識別不能
- 類似問題および模範解答の設定不備

元の分析は[analysis_report.md](analysis_report.md)を参照すること。

## 比較条件

| 条件 | 教師モデル | 生徒モデル |
| --- | --- | --- |
| Base | `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5` | Base Swallow 8B |
| v2 CoT-SFT | Base Swallow 8B + `v2_cot_sft` LoRA | Base Swallow 8B |

両条件で固定する要素は次のとおり。

- 問題セット
- 生徒モデル
- 生徒プロフィール
- 教師システムプロンプト
- 最大対話ターン数
- seed
- Judgeモデル
- 評価指標

## 問題セット

元の200問から、問題設定の矛盾、解答中の問題改変、模範解答形式の不備が検出された
18問を除外し、182問を使用する。

- 元問題数: 200
- 除外問題数: 18
- 実行問題数: 182

除外IDは
[excluded_test_question_ids.json](../shared/questions/excluded_test_question_ids.json)に記録する。
実行時の詳細な検証結果は各条件の
`data/rebuilt/<condition>/validated_questions/validation_report.json`へ保存する。

この事前検証は、明示的な矛盾と構造不備を検出するものであり、残り182問の数学的正しさを
形式的に証明するものではない。

## 実行フロー

```text
問題ペアの事前検証（200問 → 182問）
        ↓
Judge APIの認証・モデル・パラメータ疎通確認
        ↓
教師API 4基 + 固定生徒API 4基を起動
        ↓
4シャードでPhase 1対話とPhase 2解答を並列生成
        ↓
生成シャードをdialogues.jsonlへ統合
        ↓
件数・生成エラー・0ターン・情報漏洩を検証
        ↓
4シャードでGPT Judge評価
        ↓
評価シャードをevaluated_results.jsonlへ統合
        ↓
Baseとv2 CoT-SFTをrun_idで対応比較
```

## 対話生成

Phase 1では、教師と固定生徒モデルが最大10ターン対話する。生徒モデルは各ターンで
次の3項目だけを持つJSONを返す。

- `state_after`: 発話後の生徒状態
- `state_update_reason`: 状態を変更または維持した理由
- `utterance`: 教師に見せる自然な生徒発話

生徒状態には次の項目を含む。

- 理解度（0～4）
- 確信度（0～1）
- 現在の誤概念
- 感情
- 獲得済み知識
- 未習得事項

Phase 2では、Phase 1終了後のプロフィール・状態を固定生徒モデルへ渡し、類似問題を
単独で解答させる。これにより、教師との対話後に近接転移が起きたかを評価する。

## 構造化応答と情報漏洩対策

ローカルLLaMA-Factory APIは`response_format`を必ずしも強制しないため、プログラム側でも
次の検証と正規化を行う。

- 必須3項目だけを抽出し、入力を模倣した追加フィールドを破棄
- 不正なJSONバックスラッシュを限定的に補正
- 入れ子になった`utterance`から自然言語部分だけを抽出
- 状態フィールドの型・範囲・必須項目を検証
- `active_misconception`が空・欠損・非文字列の場合は直前値を維持し、正規化履歴を記録
- 理解度が1ターンで2段階以上変化する応答を拒否
- 500文字を超える生徒発話を拒否
- JSON、内部状態、`<analysis>`、`<final>`を含む発話を拒否
- 不正応答を最大3回再生成
- 再実行時は成功runを保持し、生成失敗runだけを同じ`run_id`へ原子的に置換可能

教師応答では、`<analysis>`と`<final>`を分離して保存する。終了タグが欠けていても
`<final>`開始位置以降を安全に抽出し、`<analysis>`だけで最終発話がない応答は再生成する。

## 生成完了ゲート

Judge評価へ進む前に、[validate_generation_output.py](validate_generation_output.py)で
次を検証する。

- 生成件数が検証済み問題数と一致する
- `run_id`に欠損・重複がない
- `generation_error`が0件
- Phase 1が0ターンの対話が0件
- 生徒・教師発話へのJSON、内部状態、CoTタグ露出が0件

1件でも違反があればジョブを非0で終了し、Judge評価へ進まない。

## Judge評価

Judgeモデルは既定で`gpt-5.4`を使用し、次の4評価を行う。

1. Phase 2数学正誤
2. 共感指導評価
   - 感情認識
   - 認知的共感
   - 情緒的支援
3. 数学的指導評価
   - 数学的正確性
   - 誤り診断
   - 適応的足場かけ
   - 学習確認
   - 認知負荷制御
4. 生徒らしさ評価
   - 発話の自然さ
   - 教師口調の混入
   - 知識制約違反
   - 不自然な状態更新

GPT-5.4には`max_completion_tokens`を使用する。ジョブ開始時には、モデル一覧取得だけでなく
最小のChat Completionを実行し、認証、接続、モデル名、パラメータ互換性を確認する。

評価結果は、4種類のJudgeがすべて成功した場合だけ完了扱いにする。失敗したJudgeは
再実行時にそのJudgeだけを再試行し、既存の成功結果は保持する。未取得値は0ではなく
欠測値として扱う。

## checkpoint記録

manifestおよび各生成レコードの`loaded_models`へ次を保存する。

- 教師base checkpoint
- 教師adapterの絶対パス
- 教師served model name
- 生徒checkpoint
- 生徒served model name

API起動ログでも、v2 CoT-SFT条件について`v2_cot_sft` adapterのロードを確認する。

## 並列構成とABCI資源

各条件はABCIの1ノードを使用する。

- GPU: 8基
- CPU: 192コア
- メモリ: 1,920 GB
- walltime上限: 12時間
- 生成シャード: 4
- 1シャード: 教師1 GPU + 生徒1 GPU
- Judge評価シャード: 4

## ファイル構成

主要実装は次のとおり。

| ファイル | 役割 |
| --- | --- |
| [generate_profile_update_dialogues.py](generate_profile_update_dialogues.py) | Phase 1対話とPhase 2解答の生成 |
| [evaluate_profile_update_dialogues.py](evaluate_profile_update_dialogues.py) | 4種類のJudge評価 |
| [prepare_validated_questions.py](prepare_validated_questions.py) | 問題ペアの事前検証 |
| [validate_generation_output.py](validate_generation_output.py) | Judge前の生成完了ゲート |
| [check_judge_connection.py](check_judge_connection.py) | Judge APIの実リクエスト疎通確認 |
| [analyze_comparison.py](analyze_comparison.py) | Base・SFTの対応比較レポート生成 |
| [test_rebuild.py](test_rebuild.py) | 構造化応答・再開判定等の単体テスト |
| [run_v2_condition.sh](../../jobs/test/v2/run_v2_condition.sh) | 条件共通のABCI実行フロー |

## 出力

旧テスト結果と分離するため、リビルド版は次へ保存する。

```text
SFT_abci/test/v2/data/rebuilt/
├── base_swallow8b/
│   ├── validated_questions/
│   ├── shards/
│   ├── dialogues.jsonl
│   └── evaluated_results.jsonl
└── v2_cot_sft_swallow/
    ├── validated_questions/
    ├── shards/
    ├── dialogues.jsonl
    └── evaluated_results.jsonl
```

## 実行方法

```bash
qsub SFT_abci/jobs/test/v2/run_base_swallow8b.sh
qsub SFT_abci/jobs/test/v2/run_v2_cot_sft_swallow.sh
```

両条件の正常終了後、比較レポートを生成する。

```bash
python SFT_abci/test/v2/analyze_comparison.py
```

出力は`data/rebuilt/comparison_report.md`および`comparison_report.json`となる。

## 完了判定

テスト全体の正常完了条件は次のとおり。

- 両条件で182件の対話生成が完了
- 生成エラー、0ターン、情報漏洩が0件
- 両条件で182件の4種類のJudge評価が成功
- `evaluated_results.jsonl`にrun_idの欠損・重複がない
- Baseとv2 CoT-SFTで182件の対応比較が可能

生成段階の途中結果だけでモデル性能の優劣を結論づけず、Judge評価と対応比較がすべて
完了した後に最終判断する。
