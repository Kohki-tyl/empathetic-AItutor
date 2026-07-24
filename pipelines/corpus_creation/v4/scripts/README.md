# v4 scripts

主要処理は`../run_v4.py`へ実装済みである。

1. プロフィールと独立初期感情を層化した候補対話生成
2. 6項目・60点とハード条件によるKeep／Repair／Reject監査
3. Repair対象を対話全体の文脈でまとめて修正
4. 修正済み対話の全教師ターン再監査
5. 対話単位採択とmessages JSONL変換
6. 採択経路、感情、プロフィール、完了状態、支援変更の集計
7. prompt hash、モデルrevision、実行条件による再開時検証

このフォルダーには補助的なオフライン監査スクリプトだけを置く。問題範囲とプロフィールの自動対応付けは今回の実装対象外である。

## SFT系列長監査

`audit_sft_lengths.py`は対象モデルのchat template適用後のtoken数を全件集計し、上限超過があれば終了コード2で停止する。`transformers`が必要であり、`requirements-abci.txt`で構築した環境ではvLLMの依存関係として利用できる。別環境では明示的に`transformers`を導入する。

```bash
python scripts/audit_sft_lengths.py data/run_100/v4_sft.jsonl \
  --model TARGET_BASE_MODEL \
  --revision TARGET_REVISION \
  --max-length 4096 \
  --output data/run_100/sft_length_audit.json
```

このスクリプトは自動切り詰めや自動分割を行わない。超過時はSFT側の前処理方針を決めてから再監査する。

`audit_sft_lengths.py`は対象base modelのchat templateで全SFTレコードをtokenizeし、上限超過IDをJSONへ保存する。超過があれば終了コード2となる。

```bash
python scripts/audit_sft_lengths.py data/run_100/v4_sft.jsonl \
  --model YOUR_BASE_MODEL \
  --revision YOUR_MODEL_REVISION \
  --max-length 4096
```
