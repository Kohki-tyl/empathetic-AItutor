# MathDial / EmpatheticDialogues 比較学習

研究概要の Math-SFT、Empathy-SFT、Mixed-SFT を、既存 v4 と同じ基盤モデル revision・BF16 LoRA・学習率・epoch・バッチ・最大長で実行する。Base は元モデルのまま評価する。EduEmpathy-SFT は既存 v4 を使う。混合データから教育共感コーパスを自動生成する処理ではない。

教師の原文には analysis がないため、3条件は通常の応答形式と assistant-only loss を使用する。v4 の既定の構造化応答・重み付けは維持する。したがって v4 との厳密なデータのみの比較には、v4 側も final 本文だけを使う別条件を設ける必要がある。

## 1. データの用意

公式 [MathDial](https://github.com/eth-nlped/mathdial) の `data/train.jsonl` と、公式 [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues) README のダウンロード先から取得した `train.csv`、`valid.csv` を用意する。test は入力しない。MathDial の取得 commit と両コーパスの取得元・ライセンス・取得日を実験記録に残す。入力 SHA-256 は変換時に自動保存する。

リポジトリルートから実行する。

```bash
python pipelines/sft/baselines/prepare.py mathdial --input /data/mathdial/data/train.jsonl --output /data/normalized/math_en.jsonl
python pipelines/sft/baselines/prepare.py empathy --train /data/empatheticdialogues/train.csv --valid /data/empatheticdialogues/valid.csv --output /data/normalized/empathy_en.jsonl
```

MathDial は `|EOM|`、Teacher / Student を解析し、teacher move 注釈を除く。問題文を最初の user 入力に置く。正解・誤概念注釈・プロフィールは入力に含めない。末尾の生徒発話は対応する教師ターゲットがないため除く。同じ qid は seed 固定の hash 分割で必ず同じ集合に入り、公式 test は利用しない。

EmpatheticDialogues は公式 train / valid を維持し、conv_id ごとに utterance_idx 順に復元する。最初の話者を user、聞き手を assistant とし、聞き手のみを学習する。`_comma_` を復元し、感情ラベル・状況説明をモデル入力に含めない。2コーパスで共通の system prompt を使う。

## 2. 日本語条件

日本語の機械翻訳は [TRANSLATION.md](TRANSLATION.md) を参照する。`fetch_sources.py` で公式原文から各1,000対話を seed 固定でランダム抽出し、`translate_ja.py` で GPT-6 Luna による翻訳・再開・構造検査を行う。`audit_translation.py` で原文との対応を検査し、対訳レビュー用 Markdown を作成する。人手レビュー前の出力は機械翻訳であることを明記する。

## 3. 学習入力と設定の生成

```bash
python pipelines/sft/baselines/prepare.py build --math /data/normalized/math_en.jsonl --empathy /data/normalized/empathy_en.jsonl --language en --output pipelines/sft/baselines/runs/english_full
```

`math.json/jsonl`、`empathy.json/jsonl`、`mixed.json/jsonl`、`preparation_manifest.json` が生成される。既存出力は上書きしない。

データ量を統制する場合は `--target-tokens N` を追加する。固定 revision の tokenizer で、実際に loss がかかる assistant token（終端 token を含む）を数え、対話を分割せず予算以下で抽出する。Mixed は token 予算を2コーパスへ半分ずつ配分する。各予算の95%未満しか確保できない場合は停止する。正確な同数ではないため、manifest の実測値と差を報告する。オプションなしでは全件を使い、データ量は統制されない。検証データには学習予算を適用しない。予算ありでは長さ超過で停止し、全データの最終的な長さ監査は次の preflight で行う。

## 4. 監査と学習

依存関係は `pip install -r pipelines/sft/baselines/requirements.txt`。ABCI では既存の `pipelines/sft/v4/setup_abci.sh` で作成した環境を利用できる。`baselines` と `v4` の両フォルダーが必要。

```bash
python pipelines/sft/v4/train_v4_sft.py --config pipelines/sft/baselines/runs/english_full/math.json --structure-only
python pipelines/sft/v4/train_v4_sft.py --config pipelines/sft/baselines/runs/english_full/math.json --preflight-only
python pipelines/sft/v4/train_v4_sft.py --config pipelines/sft/baselines/runs/english_full/math.json --resume auto
```

`empathy.json`、`mixed.json` も同様に実行する。各学習は同じ元モデルから独立に開始する。実学習には CUDA / BF16 対応 GPU とモデル取得が必要。暗黙の切り詰めは行わない。run manifest に設定、データhash、分割、教師token数、GPU環境を記録する。

ABCI で `baselines` から投入する場合:

```bash
qsub -P "$ABCI_GROUP" -v SFT_CONFIG=/absolute/path/to/math.json run_abci.pbs
```

比較評価は同一の日本語評価セット・生成条件で行い、数学・教育・共感指標を別々に報告する。英語のままの学習と日本語の提案コーパスの比較には言語差が残る。最良checkpointの選択には test を使用しない。

## テスト

```bash
python -m unittest discover -s pipelines/sft/baselines/tests
python -m unittest discover -s pipelines/sft/v4/tests
```

GPUやダウンロードなしで、role復元、注釈除外、assistant mask、token予算、分割漏洩検出、3条件の設定生成と監査を検証する。
