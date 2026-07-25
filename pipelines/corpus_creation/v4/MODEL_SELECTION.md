# v4コーパス作成のモデル構成

## 採用モデル

| 役割 | モデル | 実行方法 |
| --- | --- | --- |
| 生徒シミュレータ | `gpt-5.4-mini` | Chat Completions、Structured Outputs、reasoning effort `none` |
| 教師候補生成 | `gpt-5.6-terra` | Chat Completions、reasoning effort `high` |
| 初回監査 | `gpt-5.6-terra` | Batch Chat Completions、reasoning effort `high` |
| 文脈整合Repair | `gpt-5.6-terra` | Batch Chat Completions、reasoning effort `medium` |
| 修正済み対話の全ターン再監査 | `gpt-5.6-terra` | Batch Chat Completions、reasoning effort `high` |

ユーザー指定により生徒モデルは日付固定snapshotではなく`gpt-5.4-mini`エイリアスを使う。再現時にはmanifestへ実際のモデル指定、provider、SDK版、設定とプロンプトのhashを保存する。生徒は教師より軽い推論設定にし、数学問題を解く能力ではなく、E2/E3プロフィール、知識境界、感情サイクルに沿ったロールプレイを優先する。

## 役割と工程の分離

教師と生徒は別モデルにする。教師候補生成、監査、Repairは同じ`gpt-5.6-terra`を使うが、プロンプト、APIリクエスト、保存ファイルを工程ごとに分離する。

問題選定モデルは置かない。`math_train_0`から数値順に固定し、規則ベースで事前生成したE2/E3対応表からプロフィール、初期感情、誤概念を読み込む。API呼出しと実行時の選定バイアスを避ける。

Repairは対象ターン単位の独立変換ではない。同一対話のRepair対象をまとめ、対話全体を入力して前後整合性を要求する。修正後はRepairターンだけでなく、修正済み対話の全教師ターンを再監査する。

同じモデルを生成・修正・監査に使うため、数学的誤りや文体選好を共有する評価者非独立性は残る。初回Keep、Repair後Keep、Reject、数学的難問、感情困難例から層化した人手監査を別途行い、LLM Judgeとの一致を確認する必要がある。

## API方式

教師生成は前ターンに依存するため通常のChat Completionsを使う。初回監査、対話単位Repair、全ターン再監査はBatch Chat Completionsへ工程別に投入する。構造化出力で監査項目、Repair対象index、修正後教師ターンを固定する。

`gpt-5.6-terra`はChat Completions、Batch、Structured Outputs、reasoning effortに対応するため、この構成で利用できる。v4はAPI内部の推論状態を次のターンへ継承せず、監査可能な短い構造化判断をSFT教師信号へ保存する。

## 再現性

manifestへモデル名、provider、reasoning effort、SDK版、seed、prompt SHA-256、問題選択表、Batch IDを保存する。不変設定とprompt・選択表hashからfingerprintを作り、再開時に一致しなければ停止する。変更可能なのは`target_dialogues`の増加だけとし、4範囲関係を各30件に固定する`max_candidates=120`は変更しない。

## 品質管理

- 6項目すべて8点以上と全ハード条件をKeepに要求する。
- Repair対象indexの重複・不足・余分な出力を収集時に拒否する。
- 生徒発話と対象外教師発話を固定し、未来の発話とも両立するRepairだけを許す。
- Repair後は修正済み対話の全教師ターンを再監査する。
- 再監査で一つでもKeep以外なら対話全体をRejectする。
