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

現行対応表は`math_train_0`から`math_train_1097`までの欠番を除く1000問を含む。V4-S01〜V4-S08は各124〜126件である。範囲関係は`mastered` 399件、`frontier` 351件、`one_step_beyond` 200件、`far_beyond` 50件である。初期感情は`neutral` 79件、`engaged` 320件、`curious` 44件、`confused` 328件、`anxious` 179件、`frustrated` 50件である。

実際に候補生成へ使う先頭120件では、`mastered` 47件、`frontier` 43件、`one_step_beyond` 24件、`far_beyond` 6件である。`anxious`は22件、`frustrated`は6件となり、`frustrated`全件に2回の事前失敗履歴を保存する。

規則が分野根拠語を検出できなかった125件は、`curriculum_annotation.requires_human_review=true`としている。これは生成を禁止する印ではなく、正式実行前の優先的な人手確認対象である。
