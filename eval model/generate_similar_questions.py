import os
import json
import re
from openai import OpenAI
from tqdm import tqdm

api_key = os.getenv('GPT_API_KEY')
client = OpenAI(api_key=api_key)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_prompt_file(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

try:
    GENERATOR_SYSTEM = load_prompt_file(os.path.join("prompts", "similar_question_generater.txt"))
except Exception as e:
    print(f"Prompt Loading Error: {e}")
    exit(1)

# 🌟 GPT-4oの構造化出力を定義（不完全なJSONやキーのブレを完全ガード）
generator_response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "similar_question_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "similar_question": {
                    "type": "string",
                    "description": "日本語で書かれた、テキストのみで完結する類似問題"
                },
                "similar_solution": {
                    "type": "string",
                    "description": "日本語で書かれた、詳細な解説と最終的な答え"
                }
            },
            "required": ["similar_question", "similar_solution"],
            "additionalProperties": False
        }
    }
}

def is_text_only_problem(problem_text):
    if not problem_text:
        return True
    if "[asy]" in problem_text or "[/asy]" in problem_text:
        return False
    exclude_keywords = [
        r"shown.*figure", r"shown.*diagram", r"shown.*graph", r"shown.*below", r"as.*shown", r"refer.*diagram"
    ]
    for pattern in exclude_keywords:
        if re.search(pattern, problem_text, re.IGNORECASE | re.DOTALL):
            return False
    return True

def generate_similar_dataset():
    input_path = os.path.join(BASE_DIR, "questions", "test_math_questions.jsonl")
    output_path = os.path.join(BASE_DIR, "questions", "similar_math_questions.jsonl")
    
    if not os.path.exists(input_path):
        print(f"Error: 評価用元問題ファイルが見つかりません: {input_path}")
        return

    problems = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
                
    # 出力ファイルを初期化
    with open(output_path, "w", encoding="utf-8") as f:
        pass

    print(f"評価用問題 {len(problems)} 問をベースに類似問題の生成を開始します...")

    for item in tqdm(problems, desc="類似問題生成"):
        orig_q = item.get("translated_question") or item.get("problem", "")
        orig_a = item.get("translated_solution") or item.get("solution", "")
        
        prompt = f"[Original Question]\n{orig_q}\n\n[Original Solution]\n{orig_a}"
        
        try:
            response = client.chat.completions.create(
                model="gpt-5.4",  
                messages=[
                    {"role": "system", "content": GENERATOR_SYSTEM},
                    {"role": "user", "content": prompt}
                ],
                response_format=generator_response_schema,
                temperature=0.5
            )
            
            res_data = json.loads(response.choices[0].message.content)
            sim_q = res_data.get("similar_question", "")
            sim_a = res_data.get("similar_solution", "")
            
            # 生成された問題に図形参照ワードが含まれていないか最終確認
            if not is_text_only_problem(sim_q):
                continue
                
            output_data = {
                "source_id": item.get("id"),
                "type": item.get("type"),
                "level": item.get("level"),
                "original_problem": item.get("original_problem", ""),
                "original_question": orig_q,
                "similar_question": sim_q,
                "similar_solution": sim_a
            }
            
            # 1件生成ごとに即座にJSONLに追記保存
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"\nAPI Error on {item.get('id')}: {e}")
            continue

    print(f"\nすべての類似問題の生成が完了しました！\n出力先: {output_path}")

if __name__ == "__main__":
    generate_similar_dataset()