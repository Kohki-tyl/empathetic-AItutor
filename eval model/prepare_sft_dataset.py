import os
import json
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def prepare_sft_data(train_ratio=0.8, val_ratio=0.2):
    with open(os.path.join(BASE_DIR, "prompts", "teacher_system.txt"), "r", encoding="utf-8") as f:
        teacher_system = f.read()

    input_path = os.path.join(BASE_DIR, "empathetic_dialogues.jsonl")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    sessions = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                session = json.loads(line)
                # 🌟 【追加】is_completedがFalse（未完了）のセッションは学習データから除外する
                if not session.get("is_completed", False):
                    continue
                sessions.append(session)

    sft_data = []
    for session in sessions:
        problem = session.get("problem", "")
        system_content = f"{teacher_system}\n\n現在出題中の問題:\n{problem}"
        
        messages = [{"role": "system", "content": system_content}]
        
        for turn in session.get("conversation", []):
            if turn["role"] == "student":
                messages.append({"role": "user", "content": turn["content"]})
            elif turn["role"] == "teacher":
                cot_text = (
                    "<analysis>\n"
                    f"【推論プロセス】: {turn.get('thought_process', '')}\n"
                    f"【生徒の感情】: {turn.get('student_emotion', '')}\n"
                    f"&lt;ロードマップ&gt;: {turn.get('roadmap_breakdown', '')}\n"
                    f"【次の一歩の計画】: {turn.get('next_step_plan', '')}\n"
                    "</analysis>\n"
                )
                assistant_content = cot_text + turn["content"]
                messages.append({"role": "assistant", "content": assistant_content})
        
        sft_data.append({"messages": messages})

    random.seed(42)
    random.shuffle(sft_data)
    
    # 🌟 Train (8割) と Val (2割) に分割
    train_idx = int(len(sft_data) * train_ratio)
    
    train_data = sft_data[:train_idx]
    val_data = sft_data[train_idx:]

    with open(os.path.join(BASE_DIR, "sft_train.jsonl"), "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(os.path.join(BASE_DIR, "sft_val.jsonl"), "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ SFT Data Prepared (Filtered Incomplete Data) -> Train: {len(train_data)} cases, Val: {len(val_data)} cases")

if __name__ == "__main__":
    prepare_sft_data()