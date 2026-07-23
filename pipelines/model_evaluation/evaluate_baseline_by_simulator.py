import os
import json
import httpx
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# ==========================================
# 1. 基本設定と初期化
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

# 定数
MODEL_NAME = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
JUDGE_MODEL_NAME = "gpt-5.4"
MAX_TURNS = 10  # Phase 1の最大対話ターン数

# クライアントの初期化
api_key = os.environ.get('GPT_API_KEY')
openai_client = OpenAI(
    api_key=api_key,
    http_client=httpx.Client(proxy="http://proxy.abci.local:3128")
)

local_client = OpenAI(
    api_key="EMPTY", 
    base_url="http://localhost:8000/v1" 
)

# ==========================================
# 2. ヘルパー関数群
# ==========================================
def load_prompt_file(filename: str) -> str:
    """プロンプトファイルを読み込む"""
    path = BASE_DIR / "prompts" / filename
    return path.read_text(encoding="utf-8")

def load_jsonl(filename: str) -> dict:
    """JSONLファイルを読み込み、IDをキーとした辞書を返す"""
    data = {}
    path = BASE_DIR / "questions" / filename
    if not path.exists():
        return data
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item_id = item.get("id") or item.get("source_id")
                data[item_id] = item
    return data

def normalize_messages(messages: list) -> list:
    """ロールの連続を防ぐメッセージ結合関数"""
    fixed_messages = []
    for msg in messages:
        if fixed_messages and fixed_messages[-1]["role"] == msg["role"]:
            fixed_messages[-1]["content"] += "\n\n" + msg["content"]
        else:
            fixed_messages.append({"role": msg["role"], "content": msg["content"]})
    return fixed_messages

def generate_llm_response(client: OpenAI, model: str, messages: list, temperature: float, max_tokens: int = None, response_format: dict = None) -> str:
    """LLMAPI呼び出しをラップし、エラーハンドリングを共通化する"""
    kwargs = {
        "model": model,
        "messages": normalize_messages(messages),
        "temperature": temperature
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if response_format:
        kwargs["response_format"] = response_format

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()

# ==========================================
# 3. プロンプトとスキーマの読み込み
# ==========================================
TEACHER_SYSTEM = load_prompt_file("sft_teacher_system.txt")
STUDENT_SYSTEM_TEMPLATE = load_prompt_file("eval_student_system.txt")
JUDGE_SYSTEM = load_prompt_file("eval_judge_system.txt")
EMPATHY_JUDGE_SYSTEM = load_prompt_file("eval_empathy_judge_system.txt")

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

# ==========================================
# 4. メインの評価シミュレーション
# ==========================================
def run_evaluation():
    print("評価用データセットとプロファイルを読み込んでいます...")
    original_tests = load_jsonl("test_math_questions.jsonl")
    similar_tests = load_jsonl("similar_test_math_questions.jsonl")
    
    profile_path = BASE_DIR / "prompts" / "eval_student_profile.json"
    student_profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    
    output_path = BASE_DIR / "evaluation_results.jsonl"
    output_path.write_text("", encoding="utf-8")  # ファイルの初期化

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

        phase1_student_sys = STUDENT_SYSTEM_TEMPLATE.replace("{STUDENT_PROFILE}", formatted_profile).replace("{CURRENT_MODE}", "対話学習（Phase 1）")

        # コンテキストの初期化
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

        # ----------------------------------------
        # Phase 1: 対話学習セッション
        # ----------------------------------------
        for turn in range(MAX_TURNS):
            # 生徒の発話
            try:
                student_msg = generate_llm_response(local_client, MODEL_NAME, student_context, temperature=0.8, max_tokens=512)
            except Exception as e:
                tqdm.write(f"\n[Error] Student Generation Error on {q_id}: {e}")
                break
                
            dialogue_log.append({"role": "student", "content": student_msg})
            student_context.append({"role": "assistant", "content": student_msg})
            teacher_context.append({"role": "user", "content": student_msg})

            # 教師の発話
            try:
                teacher_response_str = generate_llm_response(local_client, MODEL_NAME, teacher_context, temperature=0.2, max_tokens=512)
                
                # 指導完了タグの検知とクリーンアップ
                if "[指導完了]" in teacher_response_str:
                    is_completed = True
                    teacher_msg = teacher_response_str.replace("[指導完了]", "").strip()
                else:
                    is_completed = False
                    teacher_msg = teacher_response_str

            except Exception as e:
                tqdm.write(f"\n[Error] Teacher Generation Error on {q_id}: {e}")
                break

            dialogue_log.append({"role": "teacher", "content": teacher_msg})
            student_context.append({"role": "user", "content": teacher_msg})
            teacher_context.append({"role": "assistant", "content": teacher_response_str})

            if is_completed:
                break

        # ----------------------------------------
        # Judge 1: 共感レベルの自動採点
        # ----------------------------------------
        full_dialogue_text = "\n".join([f"{d['role']}: {d['content']}" for d in dialogue_log])
        try:
            empathy_raw = generate_llm_response(
                openai_client, JUDGE_MODEL_NAME,
                [
                    {"role": "system", "content": EMPATHY_JUDGE_SYSTEM},
                    {"role": "user", "content": f"【Phase 1の対話ログ】\n{full_dialogue_text}\n\nこの対話ログを基に、教師の共感レベルと指導戦略を評価し、100点満点でスコアリングしてください。"}
                ],
                temperature=0.0, response_format=empathy_judge_response_schema
            )
            empathy_result = json.loads(empathy_raw)
        except Exception as e:
            tqdm.write(f"\n[Error] Empathy Judge Error on {q_id}: {e}")
            empathy_result = {
                "emotion_alignment_score": 0, "pedagogical_empathy_score": 0, 
                "length_control_score": 0, "total_score": 0, "empathy_reason": f"API Error: {e}"
            }

        # ----------------------------------------
        # Phase 2: 転移テスト（類似問題）
        # ----------------------------------------
        phase2_student_sys = STUDENT_SYSTEM_TEMPLATE.replace("{STUDENT_PROFILE}", formatted_profile).replace("{CURRENT_MODE}", "類似問題テスト（Phase 2）")
        phase2_prompt = f"【先ほどの学習の振り返り（参考）】\n{full_dialogue_text}\n\n【新しい出題（類似問題）】\n{sim_q}"

        try:
            student_final_answer = generate_llm_response(
                local_client, MODEL_NAME,
                [{"role": "system", "content": phase2_student_sys}, {"role": "user", "content": phase2_prompt}],
                temperature=0.2
            )
        except Exception as e:
            tqdm.write(f"\n[Error] Phase 2 Generation Error on {q_id}: {e}")
            student_final_answer = "Error during generation"

        # ----------------------------------------
        # Judge 2: 数学解答の自動採点
        # ----------------------------------------
        judge_prompt = f"【生徒の解答プロセスと最終的な答え】\n{student_final_answer}\n\n【模範解答】\n{sim_a}"
        try:
            math_judge_raw = generate_llm_response(
                openai_client, JUDGE_MODEL_NAME,
                [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": judge_prompt}],
                temperature=0.0, response_format=math_judge_response_schema
            )
            math_judge_result = json.loads(math_judge_raw)
        except Exception as e:
            tqdm.write(f"\n[Error] Math Judge Error on {q_id}: {e}")
            math_judge_result = {"is_correct": False, "judge_reason": f"API Error: {e}"}

        # ----------------------------------------
        # 結果の保存
        # ----------------------------------------
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

        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(session_result, ensure_ascii=False) + "\n")

    print(f"\n評価シミュレーション完了！ 結果は {output_path.name} に保存されました。")

if __name__ == "__main__":
    run_evaluation()