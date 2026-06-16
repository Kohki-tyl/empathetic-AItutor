import os
import json
import re
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

api_key = os.getenv('GPT_API_KEY')
client = OpenAI(api_key=api_key)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_prompt_file(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"必須のプロンプトファイルが見つかりません: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

try:
    TRANSLATOR_SYSTEM = load_prompt_file(os.path.join("prompts", "translator_system.txt"))
    print("翻訳用システムプロンプトの読み込みに成功")
except Exception as e:
    print(f"プロンプト読込エラー: {e}")
    exit(1)


def translate_dataset(limit=5):
    print("Hugging FaceからMATHデータセットを読み込み中...")

    dataset = load_dataset("nlile/hendrycks-MATH-benchmark")
    train_data = dataset["train"]
    
    print(f"読み込み完了 (総データ数: {len(train_data)}件)")
    print(f"先頭の {limit} 件の翻訳を開始\n")
    
    translated_results = []
    
    for i in tqdm(range(limit), desc="翻訳進捗"):
        item = train_data[i]

        original_problem = item.get("problem", "")
        original_solution = item.get("solution", "")
        prob_type = item.get("type", "unknown")
        prob_level = item.get("level", "unknown")
        
        prompt = f"以下の問題と解答を翻訳してください。\n\n[Original Q]\n{original_problem}\n\n[Original A]\n{original_solution}"
        
        context = [
            {"role": "system", "content": TRANSLATOR_SYSTEM},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = client.chat.completions.create(
                model="gpt-5.4", 
                messages=context,
                temperature=0.3 
            )
            translated_text = response.choices[0].message.content.strip()
            
            thought_process_match = re.search(r'\[Thought Process\](.*?)\[Q\]', translated_text, re.DOTALL)
            q_match = re.search(r'\[Q\](.*?)\[A\]', translated_text, re.DOTALL)
            a_match = re.search(r'\[A\](.*)', translated_text, re.DOTALL)
        
            thought_process_text = thought_process_match.group(1).strip() if thought_process_match else ""
            q_text = q_match.group(1).strip() if q_match else ""
            a_text = a_match.group(1).strip() if a_match else ""
            
            translated_results.append({
                "id": f"math_train_{i}",
                "type": prob_type,
                "level": prob_level,
                "original_problem": original_problem,
                "thought_process": thought_process_text,
                "translated_question": q_text,
                "original_solution": original_solution,
                "translated_solution": a_text
            })
            
        except Exception as e:
            print(f"\nAPIエラー (ID: {i}): {e}")
            continue

    output_filename = os.path.join(BASE_DIR, "translated_math.jsonl")
    
    with open(output_filename, "w", encoding="utf-8") as f:
        for res in translated_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print(f"\n翻訳処理が完了しました。結果を以下の場所に保存しました:\n   {output_filename}")


if __name__ == "__main__":
    translate_dataset(limit=200)