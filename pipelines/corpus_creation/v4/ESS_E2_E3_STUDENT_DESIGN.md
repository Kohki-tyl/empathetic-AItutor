# ESS E2・E3準拠の生徒シミュレータ設計

## 1. 根拠

Yuan et al.（2026）は、高能力LLMに部分的知識を持つ生徒を演じさせると、流暢さの背後で未習知識を使ったり、誤りが一貫しなかったりする「competence paradox」が生じると指摘している。同論文は、表面的な生徒らしさよりepistemic fidelityを優先し、知識境界、誤りの生成機構、状態遷移をEpistemic State Specification（ESS）として明示することを提案する。

本実装は次の二段階を組み合わせる。

- E2（Curriculum-Indexed）：学習済み、現在学習中、未習の範囲を明示し、外部の進行シグナルが確認された場合だけ状態を更新する。
- E3（Misconception-Structured）：問題ごとに一つの安定した誤概念または部分手続きを固定し、それが回答の誤りを因果的に決める。

実際の生徒データから状態遷移を学習・較正していないため、E4には該当しない。

## 2. プロフィール

8プロフィールは、性格ではなく教育課程位置と分野別習熟差を中心に次の認識状態を持つ。

| ID | カリキュラム位置 | 主な範囲 |
| --- | --- | --- |
| V4-S01 | 中学1年修了付近 | 基礎計算、文字式、一次方程式 |
| V4-S02 | 中学2年修了付近 | 連立方程式、一次関数、合同 |
| V4-S03 | 中学数学修了・数学I/A学習中 | 二次関数、三角比、場合の数と確率 |
| V4-S04 | 数学II/B修了・数学C/III学習中 | 高校代数、関数、数列、基礎微積分 |
| V4-S05 | 中学1年・文字式学習中 | 数の計算は既習、文字式・方程式は導入段階 |
| V4-S06 | 中学3年・代数優位 | 中学代数と関数は既習、図形は学習中 |
| V4-S07 | 中学3年・図形優位 | 中学図形は既習、確率・数え上げは学習中 |
| V4-S08 | 高校1年・数学I/A学習中 | 二次関数、三角比、場合の数と確率 |

各プロフィールには、分野別の`topic_mastery`、`curriculum_position`、`progression_signal`、`misconception_dynamics`を保存する。`prior_knowledge`は開始時に利用できる知識の完全な一覧として扱う。

## 3. 問題との事前対応

問題は`math_train_0`からIDの数値部分の昇順で固定する。実行時に問題選定LLMは呼ばない。

事前対応表は`build_problem_profile_assignments.py`で生成する。問題文と参照解答の規則ベース解析から、次を保存する。

- 数学分野
- 必要カリキュラム段階
- MATH難度
- 対応プロフィールとその分野の習得段階
- `mastered`、`frontier`、`one_step_beyond`、`far_beyond`の範囲関係
- 初期感情とその理由
- E3誤概念モデル
- 初回応答形式
- 問題内容のSHA-256

問題は数値順の先頭800件をコーパス候補、後半200件をテスト候補として分離する。それぞれの分割内で4種類の範囲関係を可能な限り均等に割り当てる。その後、各プールから`mastered`、`frontier`、`one_step_beyond`、`far_beyond`をseed 42で30件ずつ固定抽出する。コーパス側は`math_train_0`を必須問題として含める。後半200件はコーパス生成に使用しない。対応表または選択表の生成後に問題、プロフィール、規則が変更された場合はSHA-256とrun fingerprintの不一致として生成を停止する。

## 4. E3誤概念モデル

各問題には一つの`misconception_model`を割り当てる。

| フィールド | 意味 |
| --- | --- |
| `id` | 安定した誤概念の識別子 |
| `trigger` | 誤概念が現れる問題条件 |
| `faulty_procedure` | 生徒が実行する誤った部分手続き |
| `observable_signature` | 発話・途中式に現れる診断可能な特徴 |
| `repair_criterion` | 誤概念を解消してよい観察条件 |

`frontier`では分野固有の誤概念を回答の一箇所へ反映する。`mastered`では、知識不足を作為的に演じさせる代わりに、条件確認・検算の安定した省略を用いる。範囲外では、既習操作だけで着手し、最初の未習概念で停止する部分手続きを使う。

## 5. 初期感情の事前決定

初期感情はプロフィールから独立にランダム抽出せず、問題との範囲関係と難度から決定する。

| 範囲関係 | 難度条件 | 初期感情 |
| --- | --- | --- |
| mastered | level 1 | neutral |
| mastered | level 2〜3 | engaged |
| mastered | level 4〜5 | curious |
| frontier | level 1〜3 | curious |
| frontier | level 4〜5 | confused |
| one_step_beyond | level 1〜3 | confused |
| one_step_beyond | level 4〜5 | anxious |
| far_beyond | すべて | frustrated |

これはYuan et al.が直接提案した感情モデルではない。同論文のE2知識境界を本研究の学習感情へ接続するための操作的定義であり、妥当性は生成後の人手評価で確認する。

`frustrated`は単に問題が範囲外であることだけから仮定しない。`far_beyond`では教師との対話前の試行を、1回目と2回目に分けて`prior_attempt_history.attempts`へ保存する。二つの試行には、問題文から抽出した短い固有表現、異なる既習範囲内の方略、必要概念に対応する共通停止箇所を明記する。最初の生徒発話は`required_initial_disclosure`の完全一致文字列から開始し、両試行と反復した行き詰まりを教師へ明示した後、問題固有の援助を一つだけ求める。その他の範囲関係では事前失敗を捏造しない。

## 6. 現在の事前対応結果

1000問に対する`ess-e2e3-v4`規則では、V4-S01〜V4-S08は各124〜126件である。範囲関係は`mastered` 251件、`frontier` 248件、`one_step_beyond` 251件、`far_beyond` 250件となった。行列、ベクトル、二項係数、総和、複素数、床・天井関数、関数合成、二次曲線などの必要概念を検出し、プロフィールの自然言語`prior_knowledge`、現在学習中、未習範囲と照合する。

コーパス用120件では4範囲関係が各30件であり、初期感情は`neutral` 4件、`engaged` 26件、`curious` 11件、`confused` 28件、`anxious` 21件、`frustrated` 30件である。テスト用120件も4範囲関係が各30件で、初期感情は`neutral` 7件、`engaged` 23件、`curious` 17件、`confused` 23件、`anxious` 20件、`frustrated` 30件である。両選択表とも要人手確認0件、`mastered`への未習概念混入0件、範囲関係不整合0件である。これらは生成前の固定選択分布であり、採択後コーパスの分布ではない。

## 7. 限界と監査

分野・カリキュラム段階は規則ベース推定であり、教育課程の人手正解ラベルではない。特に複数分野問題、競技数学固有の発想、問題文に現れず参照解答だけに現れる手法は誤分類し得る。

したがって、`classification_confidence=conservative`、分野、範囲関係、初期感情を層化して人手確認する。`requires_human_review=true`は問題文・参照解答の欠損など機械監査を通せない場合に限定し、選択表へは含めない。生徒発話についても、指定誤概念と実際の誤りの因果的一致、未習知識の暗黙使用、修正条件前の誤概念消失をLLM Judgeと人手監査の両方で確認する。

## 参考文献

Yuan, Z., Xiao, Y., Li, M., Xuan, W., Tong, R., Diab, M., & Mitchell, T. (2026). *Towards Valid Student Simulation with Large Language Models*. arXiv:2601.05473. https://arxiv.org/abs/2601.05473
