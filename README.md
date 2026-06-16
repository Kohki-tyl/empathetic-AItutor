# Empathetic AI Tutor (empathetic-AItutor)

LLM（Large Language Models）の高度な役割演技（Role-playing）シミュレーションを利用し、学習科学および感情コンピューティングの理論に基づいた「共感的・動的足場かけ（Dynamic Scaffolding）」の数学対話データセットを自動合成するパイプラインです。

単なる正誤判定や答えを即座に提示する Solver（解答者）としてのパラダイムを超え、学習者の認知・感情状態をリアルタイムに推論し、自発的な気付きを促す次世代の「教育型LLM（Tutor）」を構築するためのSFT（教師ありファインチューニング）用コーパスの創出を目的としています。

---

## 🎓 基礎となる学術的背景とアプローチ

### 1. 開発の背景と課題
近年、LLMを活用した動的な対話型学習支援システムが注目を集めていますが、従来のシステムは単に正解や詳細な解説を一方的に出力する「Solver」としての性能に偏重しがちです。これは、学習者の理解を段階的に深める「Tutor」としての教育的知性（Pedagogical Intelligence）の実装という観点において大きな課題を残しています。
本プロジェクトでは、提示された数学の問題をあえて細分化し、生徒が自力で乗り越えられる最近接領域（ZPD: Zone of Proximal Development）のステップのみを部分的に提示する「動的足場かけ」の自動化によってこの課題を解決します。

### 2. 「認知的共感」と感情制御対話モデルの統合
教育対話においてチューターが表出するべき共感は、単なる同情や慰め（感情的共感）ではなく、学習者が「なぜそのエラーを起こしたのか」「どこで迷っているのか」を論理的に解釈し、そのつまずきを正常化（Normalize）する**「認知的共感（Cognitive Empathy）」**である必要があります。
本システムでは、D'Melloらの学習感情サイクル（Affective Learning Cycles）をベースとした11種類の動的な感情トラッキングを実装しています。生徒モデルが `Frustrated`（苛立ち）や `Bored`（退屈・思考放棄）に接近するほど、教師モデルが認知的共感を動的に強めてモチベーションを制御・維持する仕組みを構築しています。

### 3. Chain of Thought (CoT) を用いた教育的フィードバック
適切なトーンや十分な情報網羅性を備えた教育的フィードバック（FB）の生成・提供は、学習者の深い概念理解の促進において極めて有意です。
本パイプラインでは、教師モデルに強力な **Chain of Thought (CoT)** を義務付けています。発話を生成する前に、必ずシステム側で「生徒のつまずき原因のデバッグ」と「次の一歩の計画」を論理的に推論させる（メタ推論）ステップを挟むことで、極めて洗練された教育的フィードバックの自動アノテーションを実現しています。

---

## 🛠 パイプライン・アーキテクチャ

データ合成は以下の2つの独立したステップ（パイプライン）で実行されます。
```mermaid
flowchart TD
    A["[Hugging Face: MATH Benchmark]"] --> B1
    
    subgraph Step1 ["Step 1: translate_dataset.py"]
        B1["図形問題のフィルタリング<br>(Regex Parser)"]
        B2["高品質日本語翻訳<br>(CoTプロンプト)"]
        B1 --> B2
    end
    
    B2 --> C["[translated_math.jsonl]"] --> Step2
    
    subgraph Step2 ["Step 2: GM_completionsAPI.py"]
        direction TB
        S["User Simulator (Student)<br>[gpt-5.4-mini]"]
        T["Tutor Agent (Teacher)<br>[gpt-5.4]<br>└─ Structured Outputs (JSON Schema)"]
        S <-->|"マルチターン対話"| T
    end
    
    Step2 --> D["[CoT_emotional_cycle_sample.json]"]
```

### Step 1: データセットのフィルタリングと高品質日本語翻訳 (`translate_dataset.py`)
* **データソース**: Hugging Faceの `nlile/hendrycks-MATH-benchmark` を使用します。
* **フィルタリング**: テキストベースの対話シミュレーションに適さない図形問題などを、正規表現を用いて自動的に排他・フィルタリングします。
* **翻訳処理**: `translator_system.txt` のCoTプロンプトにより、数式や論理構造を維持したまま、自然な日本語の問題文・解答へと高精度に翻訳し、`translated_math.jsonl` として出力します。

### Step 2: 役割演技（Role-playing）によるマルチターン対話合成 (`GM_completionsAPI.py`)
* **User Simulator (Student Model)**:
  * `gpt-5.4-mini` を採用し、ターゲットとなる生徒のプロファイル（学年、既習範囲、苦手な領域、ミスしやすい傾向）と、問題に応じた感情遷移ルール（`student_system_v2.txt`）をインプットしてシミュレートします。
* **Tutor Agent (Teacher Model)**:
  * `gpt-5.4` を採用し、指導戦略プロンプト（`teacher_system_v2.txt`）に従って動作します。
  * OpenAI APIの **Structured Outputs (`strict: true`)** 機能を活用し、チューターの思考ログ、感情分類、および最終発話を厳密なJSON Schemaで制御・抽出します。

---

## 🎯 主要なデータ構造とスキーマ

### 1. 教師モデルの応答スキーマ (Structured Outputs)
教師モデルは毎ターン、以下のオブジェクト構造を厳密に遵守して出力を生成します。

| プロパティ名 | 型 | 説明 |
| :--- | :--- | :--- |
| `thought_process` | `string` | 生徒のつまずき原因、現在の感情状態、およびどのような足場かけが必要かのメタ推論プロセス（CoT） |
| `student_emotion` | `string` | トラッキングされた生徒の現在の感情（後述の11クラスから選択） |
| `roadmap_breakdown` | `string` | 目標達成（問題解決）までに必要なステップの分解ロードマップ |
| `next_step_plan` | `string` | このターンで提示する具体的な動的足場かけのアプローチ計画 |
| `is_completed` | `boolean` | 生徒が自力で正解に到達し、対話セッションを終了してよいかどうかのフラグ |
| `teacher_utterance` | `string` | 生徒に提示される、認认知共感と言語的足場かけを含んだ最終的な指導発話テキスト |

### 2. トラッキングされる11種類の感情クラス (`student_emotion`)
D'Melloらのモデルを拡張し、学習科学の文脈で定義された以下の11状態を追尾します。
* **ポジティブ / 促進的状態**: `Engaged`（没頭）, `Curious`（好奇心）, `Eureka`（アハ体験・突然の理解）, `Proud`（誇らしい）, `Relieved`（安心）
* **ニュートラル**: `Neutral`（中立）
* **ネガティブ / 阻害的状態（チューターの介入対象）**: `Mild_Confusion`（軽度の混乱）, `Deep_Confusion`（深い混乱）, `Frustrated`（苛立ち・フラストレーション）, `Bored`（退屈・思考放棄）, `Anxious`（不安）

---

## 💻 技術スタック

* **Language**: Python 3.12+
* **Package Manager**: `uv`（高速な仮想環境構築・依存関係同期）
* **LLM API & Infrastructure**:
  * OpenAI API (`client.chat.completions.create`)
  * 推論モデル: `gpt-5.4`（Teacher / Translator 用）, `gpt-5.4-mini`（Student Simulator 用）
  * 機能特性: Structured Outputs (`strict: true`) による構造化出力制御
* **Libraries**:
  * `datasets` (Hugging Face) : MATH ベンチマークのストリーミングロード
  * `tqdm` : 翻訳および対話シミュレーションの進捗可視化
  * `re` (標準正規表現ライブラリ) : 頑健なタグブロック（`[Thought Process]`, `[Q]`, `[A]`）の抽出

---

## 📁 ディレクトリ構成

```plaintext
empathetic-AItutor/
├── prompts/
│   ├── teacher_system_v2.txt    # 教師（チューター）の指導戦略・感情サイクル追尾プロンプト
│   ├── student_system_v2.txt    # 生徒のペルソナ・感情遷移トリガールール
│   ├── translator_system.txt    # MATHベンチマークを高精度に日本語化するCoTプロンプト
│   └── student_profile.json     # ターゲットプロファイル（学年・既習範囲・個別弱点領域の定義マスタ）
├── translate_dataset.py         # MATHデータセットのフィルタリング・日本語翻訳スクリプト
├── CoT_emotional_cycle_sample   # MATHデータセットの日本語翻訳
├── GM_completionsAPI.py         # 翻訳済みデータをシードにしたマルチターン対話合成コアスクリプト
└── README.md                    # 本ドキュメント
```

## 🚀 開発環境の構築と実行手順

### 1. 必要な環境変数
OpenAI API へのアクセスを制御するため、以下の環境変数を事前にエクスポートする必要があります。

```bash
export GPT_API_KEY="your_openai_api_key_here"
export OPENAI_BASE_URL="[https://api.openai.com/v1](https://api.openai.com/v1)"
```

### 2. セットアップ・コマンド
パッケージ管理ツール uv を用いて、高速かつクリーンに依存関係をインストールします。
```bash
pip install uv
uv pip install openai datasets tqdm
```

###3. パイプラインの実行手順
Step 1: シードデータの作成（翻訳とフィルタリング）
Hugging Face から元の MATH ベンチマークを読み込み、日本語化されたシード問題集をビルドします。
```bash
python translate_dataset.py
* 実行が成功すると、カレントディレクトリに translated_math.jsonl が生成されます。

Step 2: 共感対話シミュレーション（マルチターン合成）の実行
生成された日本語問題をシードとし、生徒プロファイルを動的にマッピングしながらチューター対話をシミュレートします。
```bash
python GM_completionsAPI.py
```
* 最大15ターンの対話がロールプレイされ、最終的に CoT_emotional_cycle_sample.json として、CoT推論ログと感情ラベルが付与された高品質なSFT用コーパスが出力されます。

## 🎯 今後の展望・研究への応用 (Future Work)
本パイプラインによって合成されたデータセットは、以下のNLP・教育工学領域におけるコア研究に直接応用可能です。

### 1 教育特化型LLMの教師ありファインチューニング（SFT）
* チューター発話（teacher_utterance）だけでなく、思考プロセス（thought_process）も含めてモデルに学習させることで、推論能力と認知的共感能力を同時に担保した実用的なAIチューターモデルの構築を目指します。

### 2 LLM-as-a-Judge による教育的フィードバックの自動評価
* 合成された対話コーパスをベースラインとし、提示された足場かけが本当に生徒モデルのZPD（最近接領域）に適合していたか、感情の悪化をどれだけ防げたかを定量評価する枠組みを構築します。

### 3 マルチモダリティ・実対話への拡張
* 現在はテキストベースのシミュレーションですが、ここで構築した感情トラッキングのロジックを、音声のトーンや韻律、あるいは数式の手書き入力ステップと統合することで、より現実の指導現場に近い、真の「共感的AI Tutor」への発展を視野に入れています。

---

## 📚 参考文献 (References)

### 和文文献 (Domestic Conferences)
* 鈴江 万碧, 堀尾 海斗, 折田 奈甫, 河原 大輔. (2025). 対話に対する共感のアノテーションと共感制御可能な対話モデルの構築. *言語処理学会 第31回年次大会 発表論文集*, pp. 4133-4136.
* 古橋 萌々香, 中山 功太, 児玉 貴志, 菅原 朔, 高見 享佑. (2026). LLM による教育的フィードバックの生成と評価. *言語処理学会 第32回年次大会 発表論文集*.
* 亀田 隆雅, 馬 青. (2025). LLM を用いた日本語学習者支援. *言語処理学会 第31回年次大会 発表論文集*.
* 井手 竜也. (2021). 生成と分類のマルチタスク学習による感情が考慮された対話応答生成. *言語処理学会 第27回年次大会 発表論文集*, pp. 643-646.

### 英文文献 (International Journals & Conferences)
* **[AutoTutor / Affective Computing]** D'Mello, S., & Graesser, A. (2012). AutoTutor and Affective AutoTutor: Learning by Talking with Cognitively and Emotionally Intelligent Computers that Talk Back. *ACM Transactions on Interactive Intelligent Systems (TiiS)*, 2(4), 1-39.
* **[KMP-Bench / Tutor Evaluation]** Shi, W., Ren, H., Pan, J., Zhou, A., Wang, K., Lu, Z., Yang, Y., Hu, Y., Wei, L., Zhan, M., & Li, H. (2026). From Solver to Tutor: Evaluating the Pedagogical Intelligence of LLMs with KMP-Bench. *arXiv preprint arXiv:2603.02775*.
* **[DialogXpert / Emotion-Aware Dialogue]** Rakib, T. B. A., Mehrish, A., Soon, L. K., Lim, W. H., & Poria, S. (2025). DialogXpert: Driving Intelligent and Emotion-Aware Conversations through Online Value-Based Reinforcement Learning with LLM Priors. *arXiv preprint arXiv:2505.17795*.
* **[Control-Value Theory]** Pekrun, R. (2024). Control-Value Theory: From Achievement Emotion to a General Theory of Human Emotions. *Educational Psychology Review*, 36(3), 83.
* **[Dialogue Tutoring System]** Perez, J., & Ong, E. (2024). Designing an LLM-Based Dialogue Tutoring System for Novice Programming. *Proceedings of the 32nd International Conference on Computers in Education (ICCE 2024)*.
* **[Pedagogical Agents]** Okonkwo, C. (2001). Affective Pedagogical Agents and User Persuasion. *Department of Computer Science, University of Saskatchewan*.
