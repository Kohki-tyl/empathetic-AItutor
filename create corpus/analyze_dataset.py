import json

def analyze_empathetic_dialogues(filepath):
    total_sessions = 0
    completed_sessions = 0
    total_turns = 0
    emotion_counts = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            total_sessions += 1
            
            if data.get("is_completed"):
                completed_sessions += 1
                
            for turn in data.get("conversation", []):
                # 教師のターンのみをカウント（1往復＝1ターンとするため）
                if turn["role"] == "teacher":
                    total_turns += 1
                    emotion = turn.get("student_emotion", "Unknown")
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    print("=== データセット統計レポート ===")
    print(f"総対話セッション数: {total_sessions} 件")
    print(f"対話終了率 (is_completed): {completed_sessions / total_sessions * 100:.1f}%")
    print(f"平均対話ターン数: {total_turns / total_sessions:.2f} ターン / 件\n")
    
    print("=== 感情ラベル出現頻度 ===")
    # 出現回数が多い順にソート
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f" - {emotion}: {count}回 ({count / total_turns * 100:.1f}%)")

if __name__ == "__main__":
    # 生成された jsonl ファイルのパスを指定して実行してください
    analyze_empathetic_dialogues("empathetic_dialogues.jsonl")