# LLM-jp・Qwen3向けv2-SFT-CoT変更点

## 目的

ABCI上に保存しているSwallow用v2-SFT-CoTスクリプトを、LLM-jpおよびQwen3でも実行できるようにするための変更点をまとめる。

学習データは次を共通して使用する。

```text
pipelines/sft/v2/data/v2_keep_only_cot_sft_train.jsonl
```

モデル間比較では、モデル固有の設定以外の学習条件を可能な限り固定する。

## 共通して維持する条件

- データセット、学習・検証分割、乱数seedを統一する
- epoch数、学習率、最大系列長、実効バッチサイズを統一する
- LoRAのrank、alpha、dropoutを統一する
- systemおよびuserのtokenを`-100`でマスクし、assistantの応答だけを損失対象にする
- assistantの`<analysis>...</analysis>`と`<final>...</final>`を両方学習対象にする
- tokenizerの特殊tokenを手作業で追加せず、各モデル固有のchat templateを使う
- モデルごとに別の出力先を使用し、checkpointを上書きしない
- 使用モデルID、revision、tokenizer設定、学習条件を実行ログに保存する

推奨する共通設定例は以下である。

```python
MAX_LENGTH = 4096
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
SEED = 42
```

QLoRAを使用する場合は、両モデルで次の量子化条件を揃える。

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

LoRAの適用対象には、モデル名を個別に列挙する代わりにPEFTの`all-linear`を使用すると両モデルを同じ条件で扱いやすい。

```python
LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)
```

## LLM-jp 7.2Bへの変更

### モデルID

```python
MODEL_NAME = "llm-jp/llm-jp-3-7.2b-instruct"
OUTPUT_DIR = "outputs/v2_cot_llm_jp_7.2b"
```

モデル情報は[Hugging Faceの公式モデルカード](https://huggingface.co/llm-jp/llm-jp-3-7.2b-instruct)を参照する。

### tokenizer

Swallow用のプロンプト文字列や特殊tokenを流用せず、LLM-jpのtokenizerに設定されたchat templateを使用する。

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
```

`pad_token`が未設定の場合だけ`eos_token`を割り当てる。

```python
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

### 主な確認事項

- Swallow固有の`bos_token`、ヘッダー文字列、role区切りを残していないこと
- SFT前に1件を整形し、system、user、assistantの順序が維持されていること
- `<analysis>`と`<final>`がchat template適用後にも残っていること
- 4096 tokenを超えた対話件数と切り詰め位置を記録すること

## Qwen3 8Bへの変更

### モデルID

```python
MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = "outputs/v2_cot_qwen3_8b"
```

モデル情報は[Hugging Faceの公式モデルカード](https://huggingface.co/Qwen/Qwen3-8B)を参照する。

### transformers

Qwen3に対応したバージョンを使用する。

```text
transformers>=4.51,<5
```

古い`transformers`ではモデル種別を認識できない場合があるため、ABCI環境のバージョンを実行前に確認する。

### thinking mode

今回のCoT教師信号は、データセット内にすでに`<analysis>`と`<final>`として記録されている。Qwen3固有のthinking形式を重ねて追加しないため、chat templateではthinking modeを無効にする。

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=False,
)
```

推論時にも同じ方針を使用する。

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

これにより、Qwen3固有のthinking出力ではなく、SFTデータで定義した次の形式を学習・評価対象にする。

```text
<analysis>
...
</analysis>
<final>
...
</final>
```

### 主な確認事項

- `enable_thinking=False`を学習時と推論時の両方で指定していること
- Qwen3固有の`<think>`とデータ側の`<analysis>`が二重に出力されていないこと
- `transformers`がQwen3対応バージョンであること
- 学習後の生成結果が`<analysis>`と`<final>`の構造を維持していること

## assistant-only lossの確認

chat templateを変えるとtoken位置も変わるため、Swallow向けに固定したtoken IDや文字列検索によるマスクは流用しない。

各モデルのchat template適用後に、以下を確認する。

1. systemとuserのlabelがすべて`-100`である
2. assistantのroleヘッダー、本文、終了tokenが学習対象である
3. padding部分のlabelが`-100`である
4. 最大系列長で切り詰めた後もassistant tokenが残っている

学習開始前に、少なくとも1件について復号したtokenとlabelマスクを表示して確認する。

## モデル読み込み例

モデルID以外は両モデルで共通化できる。

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    device_map="auto",
)
model.config.use_cache = False
```

`trust_remote_code=True`は、利用中のモデルrevisionで必要な場合だけ指定する。再現性のため、実験時にはモデルのrevisionまたはcommit hashを固定して記録する。

## 実行前チェックリスト

- [ ] LLM-jpとQwen3で同じ429件のKeep-onlyデータを使用している
- [ ] 分割後の学習件数と検証件数が一致している
- [ ] seedとデータ順が一致している
- [ ] 最大系列長と切り詰め方法が一致している
- [ ] assistant-only lossになっている
- [ ] モデル固有のchat templateを使用している
- [ ] Qwen3だけ`enable_thinking=False`を指定している
- [ ] LoRAおよびQLoRAの条件が一致している
- [ ] 出力先がモデルごとに分かれている
- [ ] 学習条件とモデルrevisionを保存している

## 評価時の注意

モデル間比較では、生徒モデル、テスト問題、Student Simulatorのプロファイル、評価プロンプト、生成条件を固定し、教師モデルだけを切り替える。Qwen3の評価時は学習時と同じく`enable_thinking=False`を使用する。
