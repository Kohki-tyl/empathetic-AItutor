import os
import json
from openai import OpenAI
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. クライアントの初期化
# 自動採点用（OpenAI API）
api_key = os.getenv('GPT_API_KEY')
openai_client = OpenAI(api_key=api_key)

# 先生役 ＆ 生徒役
local_client = OpenAI(
    api_key="EMPTY", 
    base_url="http://localhost:8000/v1" 
)

# 2. プロンプトと設定の読み込み
def load_prompt_file(filename):
    path = os.path.join(BASE_DIR, "prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

TEACHER_SYSTEM = load_prompt_file("sft_teacher_system.txt")
STUDENT_SYSTEM_TEMPLATE = load_prompt_file("eval_student_system.txt")
JUDGE_SYSTEM = load_prompt_file("eval_judge_system.txt")
EMPATHY_JUDGE_SYSTEM = load_prompt_file("eval_empathy_judge_system.txt")

MODEL_NAME = "tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3"

# 3. Judge用JSONスキーマの定義
math_judge_response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "math_judge_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_correct": {"type": "boolean"},
                "judge_reason": {"type": "string"}
            },
            "required": ["is_correct", "judge_reason"],
            "additionalProperties": False
        }
    }
}

empathy_judge_response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "empathy_judge_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "emotion_alignment_score": {"type": "integer"},
                "pedagogical_empathy_score": {"type": "integer"},
                "length_control_score": {"type": "integer"},
                "total_score": {"type": "integer"},
                "empathy_reason": {"type": "string"}
            },
            "required": [
                "emotion_alignment_score", 
                "pedagogical_empathy_score", 
                "length_control_score", 
                "total_score", 
                "empathy_reason"
            ],
            "additionalProperties": False
        }
    }
}

# 4. データ読み込み関数
def load_jsonl(filename):
    data = {}
    path = os.path.join(BASE_DIR, "questions", filename)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item_id = item.get("id") or item.get("source_id")
                data[item_id] = item
    return data

# ======== 追加: ロールの連続を防ぐ魔法のヘルパー関数 ========
def normalize_messages(messages):
    fixed_messages = []
    for msg in messages:
        if fixed_messages and fixed_messages[-1]["role"] == msg["role"]:
            fixed_messages[-1]["content"] += "\n\n" + msg["content"]
        else:
            fixed_messages.append({"role": msg["role"], "content": msg["content"]})
    return fixed_messages
# =========================================================

# 5. メインの評価シミュレーション
def run_evaluation():
    print("評価用データセットとプロファイルを読み込んでいます")
    original_tests = load_jsonl("test_math_questions.jsonl")
    similar_tests = load_jsonl("similar_test_math_questions.jsonl")
    
    profile_path = os.path.join(BASE_DIR, "prompts", "eval_student_profile.json")
    with open(profile_path, "r", encoding="utf-8") as f:
        student_profiles = json.load(f)
    
    output_path = os.path.join(BASE_DIR, "evaluation_results.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        pass

    profile_idx = 0

    for q_id, orig_item in tqdm(original_tests.items(), desc="評価シミュレーション実行中"):
        sim_item = similar_tests.get(q_id)
        if not sim_item:
            continue

        orig_q = orig_item["translated_question"]
        sim_q = sim_item["similar_question"]
        sim_a = sim_item["similar_solution"]

        # 生徒プロファイルの割り当て
        current_profile = student_profiles[profile_idx % len(student_profiles)]
        profile_idx += 1
        
        formatted_profile = (
            f"【学年】: {current_profile.get('grade', '不明')}\n"
            f"【学習済みの範囲】: {current_profile.get('learned_scope', '不明')}\n"
            f"【苦手な範囲】: {current_profile.get('weak_area', '不明')}"
        )

        # Phase 1用の生徒プロンプト構築
        phase1_student_sys = STUDENT_SYSTEM_TEMPLATE.replace("{STUDENT_PROFILE}", formatted_profile)
        phase1_student_sys = phase1_student_sys.replace("{CURRENT_MODE}", "対話学習（Phase 1）")

        # Phase 1: 対話学習セッション（最大10ターン）
        teacher_context = [
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": f"問題: {orig_q}\n\n上記の問題を出題しました。生徒の発話を待機し、対応を開始してください。"}
        ]
        
        student_context = [
            {"role": "system", "content": phase1_student_sys},
            {"role": "user", "content": f"【出題された問題】\n{orig_q}\n\nそれでは、問題文を読んで、自然に思考をスタートさせてください。"}
        ]

        dialogue_log = []
        is_completed = False

        for turn in range(10):
            # systemと問題文（先頭2つ）は常に維持し、直近6件(3往復)を残す
            if len(student_context) > 8:
                active_student_ctx = student_context[:2] + student_context[-6:]
            else:
                active_student_ctx = student_context

            # 生徒（Simulator）の発話
            try:
                res_student = local_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=normalize_messages(active_student_ctx),
                    temperature=0.8,
                    max_tokens=512
                )
                student_msg = res_student.choices[0].message.content.strip()
            except Exception as e:
                print(f"\n[Error] Student Generation Error on {q_id}: {e}")
                break
                
            # 大元のログと両者のコンテキストに生徒の発話を追加
            dialogue_log.append({"role": "student", "content": student_msg})
            student_context.append({"role": "assistant", "content": student_msg})
            teacher_context.append({"role": "user", "content": student_msg})

            # ※生徒の発話を追加した後に実行
            # systemと問題文（先頭2つ）は常に維持し、直近6件を残す
            if len(teacher_context) > 8:
                active_teacher_ctx = teacher_context[:2] + teacher_context[-6:]
            else:
                active_teacher_ctx = teacher_context

            # 教師（評価対象モデル）の発話
            try:
                res_teacher = local_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=normalize_messages(active_teacher_ctx),
                    temperature=0.2,
                    max_tokens=512
                )
                
                teacher_response_str = res_teacher.choices[0].message.content
                
                # <analysis>タグのパース処理と【対話完了判定】
                if "<analysis>" in teacher_response_str and "</analysis>" in teacher_response_str:
                    try:
                        analysis_part = teacher_response_str.split("</analysis>")[0]
                        teacher_msg = teacher_response_str.split("</analysis>")[1].strip()
                        
                        if "[指導完了]" in teacher_msg:
                            is_completed = True
                            teacher_msg = teacher_msg.replace("[指導完了]", "").strip()
                        else:
                            is_completed = False
                            
                    except Exception:
                        teacher_msg = teacher_response_str
                        is_completed = False
                else:
                    teacher_msg = teacher_response_str
                    is_completed = False

            except Exception as e:
                print(f"\n[Error] Teacher Generation Error on {q_id}: {e}")
                break

            # 発話を全てのコンテキストに記録
            dialogue_log.append({"role": "teacher", "content": teacher_msg})
            student_context.append({"role": "user", "content": teacher_msg})
            teacher_context.append({"role": "assistant", "content": teacher_response_str})

            if is_completed:
                break

        # Judge 1: 共感レベルの自動採点 (Phase 1のログを使用)
        full_dialogue_text = "\n".join([f"{d['role']}: {d['content']}" for d in dialogue_log])
        
        try:
            res_empathy = openai_client.chat.completions.create(
                model="gpt-5.4",
                messages=[
                    {"role": "system", "content": EMPATHY_JUDGE_SYSTEM},
                    {"role": "user", "content": f"【Phase 1の対話ログ】\n{full_dialogue_text}\n\nこの対話ログを基に、教師の共感レベルと指導戦略を評価し、100点満点でスコアリングしてください。"}
                ],
                response_format=empathy_judge_response_schema,
                temperature=0.0
            )
            empathy_result = json.loads(res_empathy.choices[0].message.content)
        except Exception as e:
            print(f"\n[Error] Empathy Judge Error on {q_id}: {e}")
            empathy_result = {
                "emotion_alignment_score": 0, 
                "pedagogical_empathy_score": 0, 
                "length_control_score": 0, 
                "total_score": 0, 
                "empathy_reason": f"API Error: {e}"
            }

        # Phase 2: 転移テスト（コンテキスト分離）
        phase2_student_sys = STUDENT_SYSTEM_TEMPLATE.replace("{STUDENT_PROFILE}", formatted_profile)
        phase2_student_sys = phase2_student_sys.replace("{CURRENT_MODE}", "類似問題テスト（Phase 2）")

        phase2_prompt = (
            f"【先ほどの学習の振り返り（参考）】\n{full_dialogue_text}\n\n"
            f"【新しい出題（類似問題）】\n{sim_q}"
        )

        # Phase 2のテスト解答
        try:
            res_phase2 = local_client.chat.completions.create(
                model=MODEL_NAME,
                messages=normalize_messages([
                    {"role": "system", "content": phase2_student_sys},
                    {"role": "user", "content": phase2_prompt}
                ]),
                temperature=0.2 
            )
            student_final_answer = res_phase2.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n[Error] Phase 2 Generation Error on {q_id}: {e}")
            student_final_answer = "Error during generation"

        # Judge 2: 数学解答の自動採点 (Phase 2の結果を使用)
        judge_prompt = (
            f"【生徒の解答プロセスと最終的な答え】\n{student_final_answer}\n\n"
            f"【模範解答】\n{sim_a}"
        )
        
        try:
            res_math_judge = openai_client.chat.completions.create(
                model="gpt-5.4", 
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": judge_prompt}
                ],
                response_format=math_judge_response_schema,
                temperature=0.0
            )
            math_judge_result = json.loads(res_math_judge.choices[0].message.content)
        except Exception as e:
            print(f"\n[Error] Math Judge Error on {q_id}: {e}")
            math_judge_result = {"is_correct": False, "judge_reason": f"API Error: {e}"}

        # 結果の保存
        session_result = {
            "source_id": q_id,
            "student_profile_used": current_profile,
            "phase1_turns": len(dialogue_log) // 2,
            "phase1_is_completed": is_completed,
            "phase2_student_answer": student_final_answer,
            "phase2_is_correct": math_judge_result["is_correct"],
            "math_judge_reason": math_judge_result["judge_reason"],
            "empathy_evaluation": empathy_result,
            "dialogue_log": dialogue_log
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(session_result, ensure_ascii=False) + "\n")

    print(f"\n評価シミュレーション完了 結果は {output_path} に保存されました。")

if __name__ == "__main__":
    run_evaluation()