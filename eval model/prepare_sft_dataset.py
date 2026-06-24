import os
import json
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def prepare_sft_data(train_ratio=0.8, val_ratio=0.2):
    with open(os.path.join(BASE_DIR, "prompts", "sft_teacher_system.txt"), "r", encoding="utf-8") as f:
        teacher_system = f.read().strip()

    input_path = os.path.join(BASE_DIR, "empathetic_dialogues.jsonl")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    sessions = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                session = json.loads(line)
                # 🌟 is_completedがFalse（未完了）のセッションは学習データから除外
                if not session.get("is_completed", False):
                    continue
                sessions.append(session)

    sft_data = []
    for session in sessions:
        problem = session.get("problem", "")
        
        # 🌟 変更点1: システムプロンプトに問題文を結合しない（シンプル化）
        messages = [{"role": "system", "content": teacher_system}]
        
        conversation = session.get("conversation", [])
        is_first_student_turn = True
        
        for i, turn in enumerate(conversation):
            if turn["role"] == "student":
                content = turn["content"]
                
                # 🌟 変更点2: 問題文は「生徒の最初の発話」の冒頭にコンテキストとして付与
                if is_first_student_turn:
                    content = f"[現在出題中の問題]:\n{problem}\n\n{content}"
                    is_first_student_turn = False
                    
                messages.append({"role": "user", "content": content})
                
            elif turn["role"] == "teacher":
                # 🌟 変更点3: CoTを「プロセス」「感情」「次の一歩」の3項目に極小化
                cot_text = (
                    "<analysis>\n"
                    f"【推論プロセス】: {turn.get('thought_process', '')}\n"
                    f"【生徒の感情】: {turn.get('student_emotion', '')}\n"
                    f"【次の一歩】: {turn.get('next_step_plan', '')}\n"
                    "</analysis>\n"
                )
                
                teacher_msg = turn["content"]
                
                # 🌟 変更点4: この対話の「一番最後」のターンの場合のみ、発話末尾に [指導完了] を付与
                is_last_turn = (i == len(conversation) - 1)
                if is_last_turn:
                    teacher_msg = f"{teacher_msg.strip()}\n[指導完了]"
                
                assistant_content = cot_text + teacher_msg
                messages.append({"role": "assistant", "content": assistant_content})
        
        sft_data.append({"messages": messages})

    random.seed(42)
    random.shuffle(sft_data)
    
    train_idx = int(len(sft_data) * train_ratio)
    
    train_data = sft_data[:train_idx]
    val_data = sft_data[train_idx:]

    with open(os.path.join(BASE_DIR, "sft_train.jsonl"), "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(os.path.join(BASE_DIR, "sft_val.jsonl"), "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ SFT Data Prepared (New Strategy) -> Train: {len(train_data)} cases, Val: {len(val_data)} cases")

if __name__ == "__main__":
    prepare_sft_data()