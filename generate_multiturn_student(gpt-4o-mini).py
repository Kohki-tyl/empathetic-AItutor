import os
import json
import re
from openai import OpenAI
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

try:
    TEACHER_SYSTEM = load_prompt_file("teacher_system.txt")
    STUDENT_SYSTEM_TEMPLATE = load_prompt_file("student_system.txt")
    print("プロンプトファイルの読み込みに成功")
except Exception as e:
    print(f"プロンプト読込エラー: {e}")
    exit(1)


def generate_dialogue(problem: str, profile_dict: dict, max_turns: int = 12) -> dict:

    formatted_profile = (
        f"【学年】: {profile_dict.get('grade', '不明')}\n"
        f"【学習済みの範囲】: {profile_dict.get('learned_scope', '不明')}\n"
        f"【ミスしやすい点】: {profile_dict.get('error_tendency', '不明')}\n"
        f"【苦手な範囲】: {profile_dict.get('weak_area', '不明')}"
    )
    
    student_system_prompt = STUDENT_SYSTEM_TEMPLATE.format(TARGET_PROBLEM=problem, STUDENT_PROFILE=formatted_profile)
    
    teacher_context = [
        {"role": "system", "content": TEACHER_SYSTEM},
        {"role": "user", "content": f"今から生徒が以下の問題を解き始めます。最初の出方を確認してください。\n問題: {problem}"}
    ]
    student_context = [
        {"role": "system", "content": student_system_prompt}
    ]
    
    dialogue_log = []
    is_completed = False

    for current_turn in range(max_turns):
        
        # 生徒役のターン
        try:
            if current_turn == 0:
                # 💡 初回ターン
                active_student_context = student_context + [
                    {"role": "user", "content": (
                        "【システムからの絶対指示】\n"
                        "今から提示された問題を解き始めますが、以下のルールを厳守してください。\n"
                        "1. 問題があなたの「学習済みの範囲」を超えている場合：絶対に正解を推論せず、「習っていないから分からない」と混乱するか、習っている範囲の別の知識だけで無理やり解こうとして完全に破綻した答えを出してください。\n"
                        "2. 問題が学習範囲内の場合：あなたの「ミスしやすい点」に完全に支配された誤答を提示してください。\n"
                        "いずれの場合も、絶対に自力で最初から正しいプロセスを踏まず、中学生らしい1〜2文で発話してください。"
                    )}
                ]
            else:
                active_student_context = student_context

            student_response = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=active_student_context,
                temperature=0.7
            )
            student_utterance = student_response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n生徒APIエラー: {e}")
            break
        
        dialogue_log.append({
            "turn": current_turn,
            "role": "student",
            "content": student_utterance
        })
        
        teacher_context.append({"role": "user", "content": student_utterance})
        student_context.append({"role": "assistant", "content": student_utterance})

        # 教師役のターン（GPT-5.5）
        try:
            teacher_response = client.responses.create(
                model="gpt-5.5",
                input=teacher_context,
                reasoning={"effort": "low"} 
            )
            raw_teacher_output = teacher_response.output_text
        except Exception as e:
            print(f"\n教師APIエラー: {e}")
            break
        
        emotion, error_val, roadmap_val, next_step, teacher_utterance = "", "", "", "", raw_teacher_output
        
        m_emo = re.search(r"・感情:\s*(.+)", raw_teacher_output)
        if m_emo: emotion = m_emo.group(1).strip()
            
        m_err = re.search(r"・エラー:\s*(.+)", raw_teacher_output)
        if m_err: error_val = m_err.group(1).strip()
        
        m_road = re.search(r"・ロードマップの分解:\s*(.+)", raw_teacher_output)
        if m_road: roadmap_val = m_road.group(1).strip()
            
        m_next = re.search(r"・次の一歩:\s*(.+)", raw_teacher_output)
        if m_next: next_step = m_next.group(1).strip()
            
        m_utt = re.search(r"\[教師の発話\]\s*(.*)", raw_teacher_output, re.DOTALL)
        if m_utt:
            teacher_utterance = m_utt.group(1).strip()
            teacher_utterance = teacher_utterance.replace("[教師の発話]", "").strip()
        
        dialogue_log.append({
            "turn": current_turn,
            "role": "teacher",
            "emotion": emotion,
            "error_detected": error_val,
            "roadmap_breakdown": roadmap_val,
            "next_step_plan": next_step,
            "content": teacher_utterance
        })
        
        if "[Completed]" in teacher_utterance or "[Completed]" in raw_teacher_output:
            is_completed = True
            break
        if "[Failed]" in teacher_utterance or "[Failed]" in raw_teacher_output:
            break

        student_context.append({"role": "user", "content": teacher_utterance})
        teacher_context.append({"role": "assistant", "content": raw_teacher_output})

    return {
        "problem": problem,
        "student_profile": profile_dict,
        "is_completed": is_completed,
        "conversation": dialogue_log
    }

# 実行・保存ブロック
if __name__ == "__main__":
    profile_filename = os.path.join(BASE_DIR, "student_profile.json")
    output_filename = os.path.join(BASE_DIR, "multiturn_math_student(gpt-4o-mini)_sample.json")
    
    # MATH or GSM8K を選択
    input_filename = os.path.join(BASE_DIR, "translated_math_sample.jsonl")
    
    if not os.path.exists(profile_filename):
        raise FileNotFoundError(f"エラー: プロファイルファイルが見つかりません: {profile_filename}")
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"エラー: 翻訳済み問題ファイルが見つかりません: {input_filename}")
        
    with open(profile_filename, "r", encoding="utf-8") as f:
        student_presets = json.load(f)
        
    problems_list = []
    with open(input_filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems_list.append(json.loads(line))
                
    LIMIT = 5
    target_problems = problems_list[:LIMIT]
    
    print(f"設定プロファイル: {len(student_presets)}件 をロード")
    print(f"先頭の {LIMIT} 件を使ってサンプル対話を合成\n")
    
    all_results = []
    
    for index, item in enumerate(tqdm(target_problems, desc="進捗")):
        problem_text = item.get("translated_question")
        if not problem_text:
            continue
            
        profile_item = student_presets[index % len(student_presets)]
        
        dialogue_result = generate_dialogue(problem_text, profile_item, max_turns=12)
        
        dialogue_result["source_id"] = item.get("id", f"unknown_{index}")
        
        all_results.append(dialogue_result)
        
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(json.dumps(all_results, ensure_ascii=False, indent=2))
        
    print(f"\nサンプル5件の対話データ合成が正常に完了")
    print(f"生成ファイルの保存先:\n   {output_filename}")