# 問題・プロフィール事前対応表

`problem_profile_assignments.jsonl`は、`translated_1000_math.jsonl`の`math_train_0`から数値順に1000問を並べ、各問題へ次を固定した生成前の対応表である。

- E2の問題分野・必要教育課程段階・対応プロフィール・範囲関係
- E3の誤概念または部分手続き
- 問題と学習範囲から導出した初期感情
- 初回応答形式
- 問題本文と参照解答のSHA-256

生成時に問題選定モデルやランダム抽選は使用しない。再生成はv4ディレクトリで次を実行する。

```bash
python scripts/build_problem_profile_assignments.py
```

現行対応表は`math_train_0`から`math_train_1097`までの欠番を除く1000問を含む。範囲関係は`mastered` 251件、`frontier` 248件、`one_step_beyond` 251件、`far_beyond` 250件であり、先頭800件と後半200件で均等化カウンタを分離している。

コーパス生成には、先頭800件からseed 42で固定抽出した`corpus_120_selection.json`だけを使う。`math_train_0`を必ず含み、`mastered`、`frontier`、`one_step_beyond`、`far_beyond`は各30件である。後半200件はテスト専用とし、コーパスへ使用しない。選択表は概念単位の知識境界監査結果を持ち、`mastered`への未習概念混入、範囲関係不整合、要人手確認問題が一件でもあれば再生成または実行を停止する。`far_beyond`全件には問題文の短い固有表現を含む1回目・2回目の事前試行、共通停止箇所、初回発話先頭へ完全一致で置く`required_initial_disclosure`を保存する。

`test_120_selection.json`と後半200件の`test_problem_profile_assignments.jsonl`も同じフォルダーへ同梱する。補助スクリプトの既定出力はすべてv4内部で完結し、`model_evaluation`へ書き込まない。概念規則で確定できない問題は`classification_confidence=conservative`として追跡し、空問題・空解答など実行不能なものだけを`requires_human_review=true`として選択から除外する。
