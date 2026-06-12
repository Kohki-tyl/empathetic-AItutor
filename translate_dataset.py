import os
import json
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

# 1. APIクライアントの初期化
api_key = os.getenv('GPT_API_KEY')
client = OpenAI(api_key=api_key)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_prompt_file(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"必須のプロンプトファイルが見つかりません: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# 翻訳用プロンプトのロード
try:
    TRANSLATOR_SYSTEM = load_prompt_file("translator_system.txt")
    print("翻訳用システムプロンプトの読み込みに成功")
except Exception as e:
    print(f"プロンプト読込エラー: {e}")
    exit(1)


def translate_dataset(limit=5):
    print("Hugging Faceからデータセットを読み込み中")
    
    # MATHを使用する場合
    # dataset = load_dataset("nlile/hendrycks-MATH-benchmark")
    # train_data = dataset["train"]
    
    # GSM8Kを使用する場合
    dataset = load_dataset("openai/gsm8k", "main")
    train_data = dataset["train"]
    
    print(f"読み込み完了 (総データ数: {len(train_data)}件)")
    print(f"先頭の {limit} 件の翻訳を開始\n")
    
    translated_results = []
    
    for i in tqdm(range(limit), desc="翻訳進捗"):
        item = train_data[i]
        
        # MATHのキー構造（problem / solution）
        # GSM8Kの場合は "question" と "answer" に変更
        original_problem = item.get("question", "")
        original_solution = item.get("answer", "")
        prob_type = item.get("type", "unknown")
        prob_level = item.get("level", "unknown")
        
        prompt = f"以下の問題と解答を翻訳してください。\n\n[Original Q]\n{original_problem}\n\n[Original A]\n{original_solution}"
        
        context = [
            {"role": "system", "content": TRANSLATOR_SYSTEM},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = client.responses.create(
                model="gpt-5.5",
                input=context,
                reasoning={"effort": "low"}
            )
            translated_text = response.output_text.strip()
            
            q_match = translated_text.split("[Q]")[1].split("[A]")[0].strip() if "[Q]" in translated_text else ""
            a_match = translated_text.split("[A]")[1].strip() if "[A]" in translated_text else ""
            
            translated_results.append({
                "id": f"gsm8k_train_{i}",
                "type": prob_type,
                "level": prob_level,
                "original_problem": original_problem,
                "translated_question": q_match,
                "original_solution": original_solution,
                "translated_solution": a_match
            })
            
        except Exception as e:
            print(f"\nAPIエラー (ID: {i}): {e}")
            continue

    # 保存
    output_filename = os.path.join(BASE_DIR, "translated_gsm8k_sample_2.jsonl")
    
    with open(output_filename, "w", encoding="utf-8") as f:
        for res in translated_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print(f"\n翻訳処理が完了しました 結果を以下の場所に保存しました:\n   {output_filename}")


if __name__ == "__main__":
    translate_dataset(limit=5)