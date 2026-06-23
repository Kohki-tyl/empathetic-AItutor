import os
import json
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def is_text_only_problem(item):
    problem_sources = [
        item.get("original_problem", ""),
        item.get("translated_question", ""),
        item.get("problem", ""),
        item.get("question", "")
    ]
    combined_text = "\n".join(problem_sources)
    
    if not combined_text.strip():
        return True

    if "[asy]" in combined_text or "[/asy]" in combined_text:
        return False
    
    exclude_keywords = [
        r"図.*示", r"図.*よう", r"下.*図", r"図.*参照", r"グラフ.*示", r"図.*領域", r"画.*領域",
        r"shown.*figure", r"shown.*diagram", r"shown.*graph", r"shown.*below", r"as.*shown", r"refer.*diagram"
    ]
    
    for pattern in exclude_keywords:
        if re.search(pattern, combined_text, re.IGNORECASE | re.DOTALL):
            return False
            
    return True

def filter_json_array(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    filtered_data = [item for item in data if is_text_only_problem(item)]
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
    return len(data), len(filtered_data)

def filter_jsonl(input_path, output_path):
    initial_count = 0
    filtered_count = 0
    
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            initial_count += 1
            item = json.loads(line)
            
            if is_text_only_problem(item):
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                filtered_count += 1
                
    return initial_count, filtered_count

if __name__ == "__main__":
    IS_JSONL = True  
    INPUT_FILE = "translated_math.jsonl"
    OUTPUT_FILE = "translated_math_filtered.jsonl"
    
    input_path = os.path.join(BASE_DIR, INPUT_FILE)
    output_path = os.path.join(BASE_DIR, OUTPUT_FILE)
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        exit(1)
        
    if IS_JSONL:
        init, final = filter_jsonl(input_path, output_path)
    else:
        init, final = filter_json_array(input_path, output_path)
        
    print(f"Done: {init} -> {final} (Removed: {init - final})")