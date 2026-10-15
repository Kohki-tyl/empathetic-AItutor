# EduEmpathy SFT：analysis付き1,000対話

既存の [eduempathy](../eduempathy/README.md) と同じ1,000対話、学習900／検証100を使用し、800対話版と同じプロンプト・analysis/final形式でSFTする別条件。元コーパスと通常応答版は変更していない。

## 学習内容

- `thought_process` → `<analysis>` の【認知状態】
- `student_emotion` → 【感情状態】
- `next_step_plan` → 【次の一歩】
- 正規化済み教師本文 → `<final>`。完了済み対話の最後に、800対話版と同じ `[指導完了]` を付加する。

保存済みフィールドを使用し、analysisを新規生成していない。`roadmap_breakdown` は800対話版と同様に含めない。
全3,633教師ターンが対象。ID・group_id・明示split・生徒発話・教師本文の既存正規化を維持する。

`prompts/teacher_system.txt` は `pipelines/sft/v2/prompts/v2_cot_teacher_system.txt` のコピー。
認知状態・感情状態・次の一歩の分析と、生徒向けの最終発話を出力するよう指示する。
学習条件は `research/800_vs_audited_500/config.unaudited800.json` を基準に、データ・件数・出力先を変更している。
Swallow 8Bの固定revision、BF16 LoRA、4 epochs、学習率5e-5、最大8,192 tokens、assistant-only loss。
analysisのloss重みは0.25、finalは1.0。通常応答版との比較にはプロンプト・出力形式・loss重み・完了マーカーの変更も含まれる。

## 実行

リポジトリルートから：

```bash
python -m pip install -r pipelines/sft/v4/requirements.txt
python research/corpus/sft_1000/eduempathy_analysis/train.py --structure-only
python research/corpus/sft_1000/eduempathy_analysis/train.py --preflight-only
python research/corpus/sft_1000/eduempathy_analysis/train.py --resume auto
```

実学習にはCUDA/BF16対応GPUとモデルへのアクセスが必要。
共通学習器 `pipelines/sft/v4/train_v4_sft.py` を使用するため、ABCIにはリポジトリの配置を維持してコピーする。
フォルダ内の `data.jsonl` が学習入力で、明示splitを優先する。`train.jsonl` と `validation.jsonl` は分割コピー。
出力先は本フォルダの `outputs/eduempathy_ja1000_analysis/`。同一条件で再実行するとcheckpointから再開する。
学習後のadapterは `final_adapter/` に保存される。

## ABCI

```bash
cd research/corpus/sft_1000/eduempathy_analysis
source /etc/profile.d/modules.sh
module load python/3.12/3.12.9
module load cuda/13.0/13.0.1
python -m venv .venv
source .venv/bin/activate
python -m pip install -r ../../../../pipelines/sft/v4/requirements.txt
python train.py --preflight-only
qsub -P "${ABCI_GROUP}" run_abci.pbs
```

`ABCI_GROUP` は利用するグループ名を事前に設定する。PBSは単一H200、12時間、`rt_HG` を使用する。

## 再生成と検証

```bash
python research/corpus/sft_1000/eduempathy_analysis/prepare_dataset.py
python -m unittest discover -s research/corpus/sft_1000/eduempathy_analysis/tests -v
python research/corpus/sft_1000/eduempathy_analysis/train.py --structure-only
```

再生成は隣接する通常応答版と元1,300対話、800対話版のconfig・promptを読み込む。
`manifest.json` に入力・出力hash、採択ID、分割、来歴を保存する。
analysisの制御文字は800対話版と同じく非空白C0を `$` に置換し、ANSIを除去する。DELも `$` に置換する。
変更した59フィールドは `analysis_surface_changes.json` に前後を記録している。
この処理は数式の意味を復元する保証がなく、元の通常応答版の表記正規化の制約も引き継ぐ。analysisの内容再監査は未実施。

構造・hash・900/100分割は検証済み。ローカル環境にtransformersがなく、tokenizerによる最大長監査と実学習は未実施。
analysis追加により長くなるため、GPU環境で必ずpreflightを実行する。8,192 tokens超過時は暗黙に切り詰めず停止する。
推論時に生徒へ提示するのは `<final>` 内の本文のみとする。
