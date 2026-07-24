# v4コーパス作成のモデル構成

## 採用モデル

| 役割 | モデル | 実行方法 |
| --- | --- | --- |
| 生徒シミュレータ | `tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2` | ABCI上のvLLM 0.25.1 |
| 教師候補生成 | `gpt-5.6-terra` | Chat Completions、reasoning effort `medium` |
| 初回監査 | `gpt-5.6-terra` | Batch Chat Completions、reasoning effort `high` |
| 文脈整合Repair | `gpt-5.6-terra` | Batch Chat Completions、reasoning effort `medium` |
| 修正済み対話の全ターン再監査 | `gpt-5.6-terra` | Batch Chat Completions、reasoning effort `high` |

生徒モデルはrevision `496cd5558fef4af1d426e96327d7a74681063280`へ固定する。モデルカードはvLLMで`--reasoning-parser qwen3`を使う構成を示し、Qwen3-Swallowはreasoning ON/OFF切替をサポートしないとしている。このため、現在の生徒起動設定はthinking無効化ではなくQwen3 parserを使う。[Qwen3-Swallow model card](https://huggingface.co/tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2)

## 役割と工程の分離

教師と生徒は別モデルにする。教師候補生成、監査、Repairは同じ`gpt-5.6-terra`を使うが、プロンプト、APIリクエスト、保存ファイルを工程ごとに分離する。

Repairは対象ターン単位の独立変換ではない。同一対話のRepair対象をまとめ、対話全体を入力して前後整合性を要求する。修正後はRepairターンだけでなく、修正済み対話の全教師ターンを再監査する。

同じモデルを生成・修正・監査に使うため、数学的誤りや文体選好を共有する評価者非独立性は残る。初回Keep、Repair後Keep、Reject、数学的難問、感情困難例から層化した人手監査を別途行い、LLM Judgeとの一致を確認する必要がある。

## API方式

教師生成は前ターンに依存するため通常のChat Completionsを使う。初回監査、対話単位Repair、全ターン再監査はBatch Chat Completionsへ工程別に投入する。構造化出力で監査項目、Repair対象index、修正後教師ターンを固定する。

`gpt-5.6-terra`はChat Completions、Batch、Structured Outputs、reasoning effortに対応するため、この構成で利用できる。v4はAPI内部の推論状態を次のターンへ継承せず、監査可能な短い構造化判断をSFT教師信号へ保存する。

## 再現性

manifestへモデル名、reasoning effort、student revision、期待vLLM版、seed、サンプリング設定、prompt SHA-256、Batch IDを保存する。不変設定とprompt hashからfingerprintを作り、再開時に一致しなければ停止する。`target_dialogues`と`max_candidates`の増加だけを許容する。

## 品質管理

- 6項目すべて8点以上と全ハード条件をKeepに要求する。
- Repair対象indexの重複・不足・余分な出力を収集時に拒否する。
- 生徒発話と対象外教師発話を固定し、未来の発話とも両立するRepairだけを許す。
- Repair後は修正済み対話の全教師ターンを再監査する。
- 再監査で一つでもKeep以外なら対話全体をRejectする。
