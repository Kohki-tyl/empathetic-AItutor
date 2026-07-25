# v4 SFT形式

## レコード形式

1対話を1レコードとするmessages JSONLを使う。感情変化、誤概念の持続、支援方法の変更を学習するため、通常はターン単位へ分割しない。

次のSFTの目的は、認知的・情緒的共感タスクと数学的指導タスクの両方の精度向上である。完答対話だけへ限定せず、高難度問題に対して正確で共感的な支援を継続している未完了対話も正例に含める。これにより、最終解を急いで提示する行動ではなく、生徒の状態に応じた支援過程を学習対象とする。

```text
system: v4教師system prompt
user: 問題、初期感情ラベル、最初の生徒発話
assistant: <analysis>...</analysis><final>...</final>
user: 2回目以降の生徒発話
assistant: <analysis>...</analysis><final>...</final>
```

system → user → assistantを厳密に交互配置し、連続する同一roleは変換時に拒否する。初期感情ラベルは最初のuserだけに含め、現在感情は渡さない。初回Keep対話と、文脈整合Repair後に全教師ターンの再監査を通過した対話だけを変換する。

## assistant教師信号

```text
<analysis>
【数学的評価】検算、回答分類、正しい部分、修正点
【生徒状態】理解状態、感情、生徒発話中の根拠
【支援判断】今回の一つの支援、変更理由
</analysis>
<final>
具体的な受容 → 数学的正誤 → 次の一歩
</final>
```

analysisはAPIの非公開CoTではなく、教師生成で明示的に出力した短い構造化判断記録である。`[指導完了]`は`is_completed=true`のassistant末尾へ変換処理が付与する。生成プロンプトでは予約markerの直接出力を禁止する。

## 学習設定

- systemとuserをlabel `-100`とし、assistant-only lossにする。
- assistantのanalysisとfinalを両方loss対象にする。
- 特殊tokenを手作業で追加せず、対象base modelのchat templateを使う。
- packing使用時もレコード境界、EOS、assistant maskを1件復号して確認する。
- Qwen3のthinking設定は対象checkpointのmodel cardに従う。Qwen3-Swallowはreasoning ON/OFFをサポートしないため、`enable_thinking=False`を共通設定として強制しない。

## 系列長

finalizeは文字数統計だけをmanifestへ出す。実際の可否はSFT対象モデルのtokenizerでchat template適用後に全件監査する。

1. 全レコードのtoken長と上限超過件数を記録する。
2. assistant targetを途中で切らない。
3. 自動的な右側切り詰めを行わない。
4. 超過レコードを扱う場合は、教師ターン境界でsystem、問題、必要な直近履歴を保持する前処理をSFT側で明示的に行う。

`scripts/audit_sft_lengths.py`で対象tokenizerによる全件監査を実行できる。現在のv4コーパス生成コードはtokenizer別の自動分割を実装していない。分割済みであると仮定せず、各SFTジョブで監査する。

`scripts/audit_sft_lengths.py`で全件監査できる。上限超過が一件でもあれば終了コード2となるため、SFTジョブの前段に置いて無意識な切り詰めを防ぐ。

## 評価との整合

BaseとSFTへ同じ`sft_teacher_system.txt`を与え、生徒発話の履歴追加、analysis/final parse、指導完了判定を揃える。旧test-v2の短い教師promptは主評価ではなく頑健性評価に分ける。
