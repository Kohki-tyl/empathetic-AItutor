# v4コーパス作成

v4は、数学的正確性、誤りの診断と回復、認知的共感、感情認識・情緒的支援、適応的足場かけ、理解確認・完了判定を各10点で監査し、高品質な正例を作るSFT用パイプラインである。通常Keepは6項目すべて8点以上かつ全ハード条件を満たすことを要求する。高難度問題で最大ターンに達しただけの未完了例は、数学・共感・局所支援の必須条件を満たす場合に限り`acceptable_incompleteness`として採択できる。

## 採択方針

- 初回監査ですべての教師ターンがKeepなら、その対話を採択する。
- 教師ターンだけを置換して前後の固定文脈と整合させられる場合はRepairとする。
- 同一対話のRepair対象は、対話全体・全監査結果・前後の生徒発話を入力し、まとめて修正する。
- 生徒発話とRepair対象外の教師ターンは原則変更しない。制御文字・空発話など意味内容を保持した表面復元が必要な場合だけ対象文字列を修正し、対話全体を再監査する。
- Repair後は修正済み対話の全教師ターンを再監査し、すべてKeepの場合だけ採択する。
- 初回Reject、Repair失敗、再監査で非Keepとなった対話は採択しない。再監査の具体的指摘を教師発話だけで修正できる場合は再Repairできるが、最新の対話全体監査がKeepの場合だけ採択する。
- 完了・未完了の両方を採択対象とし、train／validation分割はSFT設定側で行う。
- Good／Bad対と指導戦略ラベルは作らない。
- 次期SFTは、認知的・情緒的共感タスクと数学的指導タスクの両方の精度向上を目的とする。
- 高難度問題で正確かつ共感的な支援を継続している場合、10ターン以内に最終解・検算・理由説明へ到達しないことや、足場が細かくなったことだけでは非Keepにしない。

現行要件の一覧は[V4_CORPUS_REQUIREMENTS.md](./V4_CORPUS_REQUIREMENTS.md)を正本とする。厳密な採択条件は[ADOPTION_CRITERIA.md](./ADOPTION_CRITERIA.md)、モデル仕様は[MODEL_SELECTION.md](./MODEL_SELECTION.md)、SFT形式は[SFT_FORMAT.md](./SFT_FORMAT.md)、E2/E3生徒設計は[ESS_E2_E3_STUDENT_DESIGN.md](./ESS_E2_E3_STUDENT_DESIGN.md)、全体の妥当性は[V4_CORPUS_DESIGN_AND_FEASIBILITY.md](./V4_CORPUS_DESIGN_AND_FEASIBILITY.md)を参照する。

## 実行環境

教師生成は`gpt-5.6-terra`の通常Chat Completionsを使う。監査・Repair・再監査は原則Batch Chat Completionsとし、Batchファイル経路が利用できない場合は、通常Chat Completionsで1対話を1リクエストとして監査する。生徒役はOpenAI APIの`gpt-5.4-mini`（reasoning effort `none`）とし、教師役とモデルを分離する。生徒出力にもStructured Outputsを使い、ターンごとの感情遷移と応答段階をJSON Schemaで制約する。

```bash
cd v4
module load python/3.12/3.12.9
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

`.env`へ`OPENAI_API_KEY`を設定する。OpenAI生徒ではGPU、CUDA、`STUDENT_BASE_URL`は不要である。ABCIグループはPBSへ埋め込まず、`qsub -P "$ABCI_GROUP" abci/generate_dialogues.pbs`で指定する。10件だけ試す場合は`python run_v4.py generate --config configs/pilot10.openai.json --limit 10`を使い、本番出力と分離する。用途限定の設定は[configs/](./configs/)へまとめている。

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

Batchが利用できない場合の対話全体監査は次を使う。生徒の内部メタデータだけの不整合は`metadata_warnings`へ分離され、教師SFTの採択を妨げない。

```bash
python run_v4.py audit-dialogues-sync --workers 4
```

対象0件のRepair／再監査は`skipped`として安全に処理される。Batchが未完了なら、完了後に同じ`collect-*`を再実行する。通常の再開時に`--overwrite`は付けない。

ABCIの生成ジョブは`rt_HG`の12時間上限に合わせている。対話生成が時間内に終わらない場合は、同じ`config.json`でジョブを再投入する。候補番号と問題番号の対応は固定され、既存の正常候補を生成し直さない。

## 再現性と再開

manifestへ、変更不可設定、モデル、provider、reasoning effort、seed、プロンプトSHA-256、実行環境、Batch ID、run fingerprintを保存する。再開時にfingerprintが変わっていれば処理を停止する。

`target_dialogues`だけは採択目標の変更用として履歴をmanifestへ残す。`max_candidates=120`は、4種類の範囲関係を各30件とする固定選択表と一体なので変更しない。最初から別条件で実行する場合は選択表を再作成し、別の`output_dir`を使うか、意図を確認したうえで`generate --overwrite`を使う。

## 生徒条件

V4-S01〜V4-S08はESSのE2とE3を組み合わせる。学年段階に加えて、代数優位、図形優位、確率学習中など分野別習得段階の異なるプロフィールを持つ。E2として学習済み・現在学習中・未習の範囲と進行シグナルを、E3として問題ごとのtrigger、faulty procedure、observable signature、repair criterionを固定する。`prior_knowledge`を開始時の使用可能知識の完全な一覧とし、生徒は各発話で`response_stage`と`knowledge_used`を返す。

初期感情はランダム抽出せず、問題の必要範囲とプロフィールの分野別習得段階の関係、およびMATH難度から事前決定する。最初の生徒発話では、その感情を語調・ためらい・確信の強さへ反映する。理解度は1ターン最大1、確信度は最大0.25、感情は定義済みサイクルの隣接状態だけ変化できる。初回だけは教師介入前なので、理解度、獲得知識、未習範囲、誤概念を変更できず、確信度変化も0.1以内とする。

問題選定LLMは使用しない。全1000問を数値順に並べ、先頭800問をコーパス候補プール、後半200問をテスト専用プールとして交差を禁止する。行列、ベクトル、二項係数、総和、複素数、床・天井関数、関数合成、二次曲線などを概念規則で検出し、プロフィールの自然言語`prior_knowledge`と照合して実効習熟度を決める。コーパス生成では`assignments/corpus_120_selection.json`を参照し、`math_train_0`を必ず含めたうえで、先頭800問から4種類の範囲関係をseed 42で各30件、計120件選ぶ。選択時と実行時の両方で、`mastered`への未習概念混入、範囲関係不整合、要人手確認問題を拒否する。

初回生徒発話も対応表で固定する。`frontier`では指定誤概念に基づく誤答または部分手続き、`mastered`では検算・条件確認が一箇所不足した回答、範囲外では具体的援助要請を生成する。無関係なランダム誤答は作らない。

教師には初回だけ、`scope_relation`、開始時の既習・未習範囲、初期感情、事前試行履歴を渡す。現在感情や対話中の獲得知識は構造化条件として渡さず、2ターン目以降は生徒発話と対話履歴から教師が判断する。不安や反復した行き詰まりは着眼点の承認と分けて具体的に受容し、問いだけでなく例・記号・公式も使用可能知識と照合する。未習内容が必要なら、平易な意味づけの後に問題本体より小さい再現・識別課題を一つだけ提示する。

教師の`is_completed=true`は、最新回答が理由を含む正答で、`next_support=なし`、追加質問なしの場合だけ許可する。生徒・教師の構造検証に失敗した場合は最大3回まで検証理由を返して再生成する。不正出力そのものと検証理由は`generation_diagnostics`へ保存する。

獲得知識はPython側の状態を正本とする。生徒モデルには累積済み`state_before.acquired_knowledge`を入力するが、出力させるのは今回増えた`newly_acquired_knowledge`だけである。検証後にPython側で順序を保って累積するため、長い全件再出力による既習知識の脱落を防ぐ。

## 出力

候補対話、監査、Repair、採択コーパスはrun別に`data/`へ保存する。現在の最終104件は、履歴上の名前を維持した`data/run_10_openai_gpt54mini/`にある。旧pilot、分割生成run、最終成果物を混在させない。詳細は[data/README.md](./data/README.md)を参照する。

SFT出力は問題、初期感情ラベル、最初の生徒発話を同一userに入れ、roleを交互にする。Base・SFTのtest-v4評価でも教師へ初回だけ同じ初期感情ラベルを渡す。assistantは短い監査可能な`<analysis>`と生徒向け`<final>`を学習対象にする。使用モデルのtokenizerで全件の系列長を確認し、assistant targetを途中で切り詰めない。
