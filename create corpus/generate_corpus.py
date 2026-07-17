import os
import json
from openai import OpenAI
from tqdm import tqdm

api_key = os.getenv('GPT_API_KEY')
client = OpenAI(api_key=api_key)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_prompt_file(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

try:
    TEACHER_SYSTEM = load_prompt_file(os.path.join("prompts", "teacher_system.txt"))
    STUDENT_SYSTEM_TEMPLATE = load_prompt_file(os.path.join("prompts", "student_system.txt"))
except Exception as e:
    exit(1)

teacher_response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "teacher_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thought_process": {"type": "string"},
                "student_emotion": {
                    "type": "string",
                    "enum": [
                        "Engaged", "Curious", "Neutral", "Confusion", 
                        "Frustrated", "Bored", "Anxious", "Eureka", "Proud", "Relieved"
                    ]
                },
                "roadmap_breakdown": {"type": "string"},
                "next_step_plan": {"type": "string"},
                "is_completed": {"type": "boolean"},
                "teacher_utterance": {"type": "string"}
            },
            "required": [
                "thought_process", "student_emotion", "roadmap_breakdown", 
                "next_step_plan", "is_completed", "teacher_utterance"
            ],
            "additionalProperties": False
        }
    }
}

def generate_dialogue(problem: str, profile_dict: dict, initial_condition: str, max_turns: int = 15) -> dict:
    formatted_profile = (
        f"【学年】: {profile_dict.get('grade', '不明')}\n"
        f"【学習済みの範囲】: {profile_dict.get('learned_scope', '不明')}\n"
        f"【ミスしやすい点】: {profile_dict.get('error_tendency', '不明')}\n"
        f"【苦手な範囲】: {profile_dict.get('weak_area', '不明')}"
    )
    
    student_system_prompt = STUDENT_SYSTEM_TEMPLATE.format(
        TARGET_PROBLEM=problem, 
        STUDENT_PROFILE=formatted_profile,
        INITIAL_CONDITION=initial_condition
    )
    
    teacher_context = [
        {"role": "system", "content": TEACHER_SYSTEM},
        {"role": "user", "content": f"問題: {problem}\n\n上記の問題を出題しました。生徒の最初の発話を待機し、対応を開始してください。"}
    ]
    student_context = [
        {"role": "system", "content": student_system_prompt}
    ]
    
    dialogue_log = []
    is_completed = False

    for current_turn in range(max_turns):
        try:
            if current_turn == 0:
                active_student_context = student_context + [{"role": "user", "content": "それでは、提示された問題を解き始めてください。"}]
            else:
                active_student_context = student_context

            student_response = client.chat.completions.create(
                model="gpt-5.4-mini", 
                messages=active_student_context,
                temperature=0.8 
            )
            student_utterance = student_response.choices[0].message.content.strip()
        except Exception as e:
            break
        
        dialogue_log.append({
            "turn": current_turn,
            "role": "student",
            "content": student_utterance
        })
        
        teacher_context.append({"role": "user", "content": student_utterance})
        student_context.append({"role": "assistant", "content": student_utterance})

        try:
            teacher_response = client.chat.completions.create(
                model="gpt-5.4", 
                messages=teacher_context,
                response_format=teacher_response_schema,
                temperature=0.2
            )
            
            teacher_data = json.loads(teacher_response.choices[0].message.content)
            thought_process = teacher_data.get("thought_process", "")
            student_emotion = teacher_data.get("student_emotion", "")
            roadmap_val = teacher_data.get("roadmap_breakdown", "")
            next_step = teacher_data.get("next_step_plan", "")
            is_completed = teacher_data.get("is_completed", False)
            teacher_utterance = teacher_data.get("teacher_utterance", "")
            
        except Exception as e:
            print(f"Teacher API Error: {e}")
            break
        
        dialogue_log.append({
            "turn": current_turn,
            "role": "teacher",
            "thought_process": thought_process,
            "student_emotion": student_emotion,
            "roadmap_breakdown": roadmap_val,
            "next_step_plan": next_step,
            "content": teacher_utterance
        })
        
        if is_completed:
            break

        student_context.append({"role": "user", "content": teacher_utterance})
        teacher_context.append({"role": "assistant", "content": teacher_utterance})

    return {
        "problem": problem,
        "student_profile": profile_dict,
        "is_completed": is_completed,
        "conversation": dialogue_log
    }

if __name__ == "__main__":
    profile_filename = os.path.join(BASE_DIR, "prompts", "student_profile.json")
    output_filename = os.path.join(BASE_DIR, "800_empathetic_dialogues.jsonl")
    input_filename = os.path.join(BASE_DIR, "questions", "translated_1000_math.jsonl")
    
    with open(profile_filename, "r", encoding="utf-8") as f:
        student_presets = json.load(f)
        
    problems_list = []
    with open(input_filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems_list.append(json.loads(line))
                
    LIMIT = 500
    target_problems = problems_list[:LIMIT]
    
    condition_presets = [
        "あなたは現在【Frustrated】状態です。気分の悪さ、あるいはこの問題への強い苦手意識により、最初からイライラしており投げやりなトーンです。",
        "あなたは現在【Confusion】状態です。問題を見て少し戸惑っており、自信がなさそうに解き始めます。",
        "あなたは現在【Engaged】状態です。前向きに解く意欲があり、集中して取り組み始めます。"
    ]
    
    # --- レジューム機能 ---
    start_index = 0
    if os.path.exists(output_filename):
        with open(output_filename, "r", encoding="utf-8") as f:
            start_index = sum(1 for line in f if line.strip())
        print(f"\n既存のデータを {start_index} 件検出しました。続きから生成を再開します...")
    else:
        with open(output_filename, "w", encoding="utf-8") as f:
            pass
            
    remaining_problems = target_problems[start_index:]
    
    # 生成ループ
    for i, item in enumerate(tqdm(remaining_problems, initial=start_index, total=LIMIT)):
        real_index = start_index + i
        
        problem_text = item.get("translated_question")
        if not problem_text:
            continue
            
        profile_item = student_presets[real_index % len(student_presets)]
        selected_condition = condition_presets[real_index % len(condition_presets)]
        
        # 1. 対話を生成
        dialogue_result = generate_dialogue(problem_text, profile_item, selected_condition, max_turns=15)
        
        # 2. 元のソースコード通りに source_id を付与
        dialogue_result["source_id"] = item.get("id", f"unknown_{real_index}")
        
        # 3. 元のフォーマットのまま追記保存
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(dialogue_result, ensure_ascii=False) + "\n")