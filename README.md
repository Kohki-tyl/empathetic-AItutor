# Empathetic AI Tutor (empathetic-AItutor)
---
## プロジェクトの概要
本プロジェクトは、LLM（Large Language Models）の高度な相互作用（User Simulator ⇄ Tutor Agent）を利用し、学習科学および感情コンピューティングの理論に基づいた高品質な「共感的・動的足場かけ（Dynamic Scaffolding）」の数学対話データセットを自動合成するパイプラインです。<br>
単なる正誤判定や答えの提示（Solver）を超え、学習者の認知・感情状態に寄り添いながら自発的な気付きを促す次世代の「教育型LLM（Tutor）」を構築するためのSFT（教師ありファインチューニング）用コーパス創出を目的としています。本システムは、国内外の主要な対話システム、感情制御、および教育支援に関する以下の学術的知見を背景に設計されています。<br>
### 🎓 基礎となる学術的背景とアプローチ
* LLMを用いた教育支援の実用化と課題
  * 近年、LLMを活用した言語学習支援やプログラミング教育など、動的な対話を通じた学習者支援システムの開発が急速に進展しています。
  * しかし、従来のLLMは単に正解を出力する「Solver（解答者）」としての性能に偏重しがちであり、学習者の理解を段階的に深める「Tutor（指導者）」としての教育的知性（Pedagogical Intelligence）の評価や実装が大きな課題となっています。
  * 本プロジェクトでは、問題をあえて細分化し、生徒が自力で乗り越えられる最近接領域（ZPD）のステップのみを提示する「動的足場かけ」の自動化によってこの課題を解決します。
* 「認知的共感」と感情制御対話モデルの統合
  * オープンドメインや雑談対話の分野では、ユーザーの感情（表出感情・経験感情）を高度に認識・分類し、共感的な応答を制御するためのコーパス構築やマルチタスク学習の研究が盛んに行われています。
  * 特に教育対話においてチューターが表出する共感は、単なる同情（感情的共感）ではなく、学習者が「なぜそのエラーを起こしたのか」「どこで迷っているのか」を論理的に解釈し、そのつまずきを正常化（Normalize）する「認知的共感（Cognitive Empathy）」である必要があります。
  * 本システムでは、D'Melloらの学習感情サイクルをベースとした11種類の動的な感情トラッキングに加え、生徒モデルが Frustrated（苛立ち）や Bored（退屈・思考放棄）に接近するほど、教師モデルが認知的共感を動的に強めてモチベーションを制御する仕組みを実装しています。
* 教育的フィードバック（Pedagogical Feedback）の有意さ
  * STEM教育をはじめとする学習支援において、適切なトーンや十分な情報網羅性を備えた教育的フィードバック（FB）を生成・提供することは、学習者の深い概念理解の促進において極めて有意であることが実証されています。
  * また、教育的な対話生成においては、表面的なテキストの一致だけでなく、教育的特徴を反映したLLM-as-a-Judge等の評価枠組みを回すことがデータの品質管理において有効です。
  * 本パイプラインでは、教師モデルに強力な Chain of Thought (CoT) を義務付け、発話を生成する前に必ず「つまずき原因のデバッグ」と「次の一歩の計画」を論理的に推論させることで、極めて洗練された教育的フィードバックの自動アノテーションを実現しています。

## 使用している主な技術
分類技術・ライブラリ用途 / 特徴LanguagePython 3.12+プログラミング言語LLM APIOpenAI APIclient.chat.completions.create (Structured Outputs / strict: true)Package Manageruv高速な仮想環境構築・依存関係管理Dataset SourceHugging Face datasetsnlile/hendrycks-MATH-benchmark のロード用Utilitytqdm翻訳・生成処理の進捗バー表示Regex Parserre (標準ライブラリ)re.DOTALL による頑健なタグブロック抽出と図形問題のフィルタリング

## ディレクトリ構成
Plaintextempathetic-AItutor/
├── prompts/
│   ├── teacher_system.txt       # 教師（チューター）の指導戦略・感情サイクル追尾プロンプト
│   ├── student_system.txt       # 生徒のペルソナ・感情遷移トリガールール
│   └── translator_system.txt    # MATHベンチマークを高精度に日本語化するCoTプロンプト
├── student_profile.json         # 中1〜中3のターゲットプロファイル（学年・既習範囲・弱点領域）
├── translate_dataset.py         # MATHデータセットのフィルタリング・日本語翻訳スクリプト
├── generate_dialogue.py        # 翻訳済み問題をシードにしたマルチターン対話合成コアスクリプト
└── README.md                    # 本ドキュメント

## 必要な環境変数やコマンド一覧
1. 必要な環境変数APIアクセスおよび動作モデルの制御のため、以下の環境変数を設定する必要があります。変数名設定値の例説明GPT_API_KEYsk-proj-xxxx...OpenAIのAPIキー（必須）OPENAI_BASE_URLhttps://api.openai.com/v1OpenAI APIのエンドポイント（必須）
2. コマンド一覧実行コマンド役割pip install uvパッケージ管理ツール uv のグローバルインストールuv pip install openai datasets tqdmプロジェクトに必要な依存ライブラリの一括インストールpython translate_dataset.pyMATHデータセットから図形問題を自動排他し、日本語に翻訳（Step 1）python generate_dialogue.py翻訳データをシードに、生徒と教師のマルチターン対話を合成（Step 2）

## 開発環境の構築方法
以下のステップ順に実行することで、クリーンな開発環境を即座に構築できます。
Step 1: リポジトリのクローンと移動
Bash
git clone https://github.com/Kohki-tyl/empathetic-AItutor.git
cd empathetic-AItutor
Step 2: 仮想環境の作成と依存関係のインストールパッケージ管理ツール uv を使用して環境を構築します。
Bash
1. uvのインストール
pip install uv
2. 依存関係のインストール（仮想環境の作成から同期まで自動で行われます）
uv pip install openai datasets tqdm
Step 3: 環境変数の設定~/.bashrc または ~/.zshrc に以下の行を追加し、source コマンドでターミナルに反映させてください。
Bash
APIキーとBase URLの登録
export GPT_API_KEY="あなたのOpenAI_APIキー"
export OPENAI_BASE_URL="https://api.openai.com/v1"
設定の反映
source ~/.bashrc
Step 4: パイプラインの動作確認環境構築が完了したら、以下のコマンドを順に実行してデータ合成のテスト（初期設定では少数件数）を行ってください。
Bash
データセットのフィルタリング & 翻訳の実行
python translate_dataset.py
共感対話シミュレーション（マルチターン生成）の実行
python generate_dialogue.py
それぞれ translated_math_sample_CoT.jsonl と CoT_emotional_cycle_sample.json がディレクトリ内に正常に生成されれば、開発環境の構築はすべて成功です。
