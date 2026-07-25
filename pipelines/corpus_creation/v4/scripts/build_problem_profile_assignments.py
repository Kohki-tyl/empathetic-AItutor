"""問題、E2/E3プロフィール、初期感情の事前対応表を決定的に作成する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = BASE_DIR.parent / "questions" / "translated_1000_math.jsonl"
DEFAULT_PROFILES = BASE_DIR / "prompts" / "student_profiles.json"
DEFAULT_OUTPUT = BASE_DIR / "assignments" / "problem_profile_assignments.jsonl"
POLICY_VERSION = "ess-e2e3-v1"
MATH_TRAIN_ID = re.compile(r"^math_train_(\d+)$")

TOPIC_RULES: dict[str, list[str]] = {
    "calculus": [
        r"derivative", r"differentiat", r"integral", r"微分", r"積分",
        r"\\int", r"\\frac\{d",
    ],
    "trigonometry": [
        r"\\sin", r"\\cos", r"\\tan", r"trigon", r"三角比", r"三角関数",
        r"正弦定理", r"余弦定理",
    ],
    "probability_combinatorics": [
        r"probability", r"確率", r"permutation", r"combination", r"場合の数",
        r"順列", r"組合せ", r"\\binom", r"arrang", r"ways?\b", r"通り",
    ],
    "number_theory": [
        r"prime", r"素数", r"divisor", r"約数", r"divisible", r"倍数",
        r"remainder", r"余り", r"modulo", r"合同式", r"gcd", r"lcm",
        r"最大公約数", r"最小公倍数", r"base[- ]?\d", r"進法",
    ],
    "functions_sequences": [
        r"sequence", r"数列", r"progression", r"等差", r"等比", r"logarith",
        r"対数", r"exponential", r"指数関数", r"complex", r"複素数",
        r"asymptote", r"漸近線", r"function", r"関数",
    ],
    "geometry": [
        r"triangle", r"三角形", r"circle", r"円", r"angle", r"角", r"polygon",
        r"多角形", r"area", r"面積", r"volume", r"体積", r"radius", r"半径",
        r"diameter", r"直径", r"coordinate plane", r"座標平面", r"perimeter", r"周",
    ],
    "algebra": [
        r"equation", r"方程式", r"polynomial", r"多項式", r"factor", r"因数",
        r"inequal", r"不等式", r"roots?\b", r"解を求", r"simplif", r"整理",
        r"[a-z]\s*[=<>]", r"[a-z]\^2",
    ],
}
TOPIC_SPECIFICITY_BONUS = {
    "calculus": 5, "trigonometry": 5, "probability_combinatorics": 4,
    "number_theory": 3, "functions_sequences": 3, "geometry": 2,
    "algebra": 0,
}

BASE_REQUIRED_STAGE = {
    "arithmetic": 1,
    "algebra": 1,
    "geometry": 2,
    "probability_combinatorics": 2,
    "number_theory": 2,
    "functions_sequences": 3,
    "trigonometry": 3,
    "calculus": 4,
}

SCOPE_LABELS = {
    1: "中学1年までの基礎計算・文字式・一次方程式",
    2: "中学校数学修了相当",
    3: "高校数学I・A相当",
    4: "高校数学II・B・Cまたは発展・競技数学相当",
}

TOPIC_MISCONCEPTIONS = {
    "arithmetic": {
        "id": "M-ARITH-ORDER",
        "label": "演算順序または符号を局所的に取り違える",
        "trigger": "複数の演算、負号、分数が同じ式に現れる",
        "faulty_procedure": "左から見えた演算を先に処理し、符号と括弧の作用域を再確認しない",
        "observable_signature": "途中の一箇所だけ演算順序または符号が変わる",
        "repair_criterion": "演算順序と符号の作用域を自分の言葉で確認し、同型の一段階を正しく実行する",
    },
    "algebra": {
        "id": "M-ALG-EQUIV",
        "label": "等式変形で両辺へ同じ操作を適用し損ねる",
        "trigger": "移項、分配、分母除去を含む方程式を処理する",
        "faulty_procedure": "項を反対側へ移すことだけを覚え、等価性を保つ操作として追跡しない",
        "observable_signature": "符号変更または係数処理が一箇所だけ不整合になる",
        "repair_criterion": "両辺へ行った操作を明示し、元の式への代入確認まで行う",
    },
    "geometry": {
        "id": "M-GEO-DIAGRAM",
        "label": "図の見た目から未提示の等長・垂直・相似を仮定する",
        "trigger": "補助線や複数の図形関係を含む幾何問題に取り組む",
        "faulty_procedure": "図でそう見える関係を条件または定理で確認せず使用する",
        "observable_signature": "根拠のない等長、直角、平行、相似のいずれかを置く",
        "repair_criterion": "使用する関係を問題の条件または既習定理へ結び付けて説明する",
    },
    "probability_combinatorics": {
        "id": "M-PROB-SAMPLE",
        "label": "標本空間の等確率性または重複を確認しない",
        "trigger": "場合分け、並べ方、複数段階の選択を数える",
        "faulty_procedure": "見つけた場合の数をそのまま足し、重複と全事象の等確率性を確認しない",
        "observable_signature": "分母と分子で異なる数え方を使うか、場合の重複が生じる",
        "repair_criterion": "一つの標本を定義し、全事象と条件を満たす事象を同じ単位で数える",
    },
    "number_theory": {
        "id": "M-NUM-PATTERN",
        "label": "少数の例で見えた整数パターンを一般化する",
        "trigger": "約数、余り、素数、桁に関する一般的結論を求める",
        "faulty_procedure": "小さい数で成立した規則を必要条件・十分条件の確認なしに採用する",
        "observable_signature": "例示はあるが任意の整数に対する根拠がない",
        "repair_criterion": "割り算の式、因数分解、合同関係のいずれかで一般の場合を説明する",
    },
    "functions_sequences": {
        "id": "M-FUNC-LOCAL",
        "label": "局所的な値の変化を一般式の性質と混同する",
        "trigger": "関数、数列、指数・対数の全体的性質を判断する",
        "faulty_procedure": "代入した少数の値だけから増減、周期、一般項を決める",
        "observable_signature": "具体例は正しいが一般式または定義域の確認がない",
        "repair_criterion": "式の構造または定義を用いて対象範囲全体へ理由を拡張する",
    },
    "trigonometry": {
        "id": "M-TRIG-RATIO",
        "label": "角、辺、三角比の対応を固定せず公式へ代入する",
        "trigger": "複数の角または辺を含む三角比・三角関数を使用する",
        "faulty_procedure": "対辺・隣辺または対応角を確認せず、覚えている式へ数値を入れる",
        "observable_signature": "式の形は近いが角と辺の対応が入れ替わる",
        "repair_criterion": "基準角と各辺の役割を図または言葉で対応付けてから式を立てる",
    },
    "calculus": {
        "id": "M-CALC-SYMBOL",
        "label": "微分・積分の記号操作と量の意味を切り離す",
        "trigger": "変化率、接線、面積と微積分の式を結び付ける",
        "faulty_procedure": "記号操作だけを実行し、定義域、定数、求める量との対応を確認しない",
        "observable_signature": "微分式または積分式は書けるが何を表すか説明できない",
        "repair_criterion": "得られた式を元の変化率・接線・面積へ戻して解釈する",
    },
}

RELATION_ORDER = ["mastered", "frontier", "one_step_beyond", "far_beyond"]
RELATION_TARGET_WEIGHTS = {
    "mastered": 40,
    "frontier": 35,
    "one_step_beyond": 20,
    "far_beyond": 5,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ordered_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numbered: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        source_id = str(row.get("id") or row.get("source_id") or "")
        match = MATH_TRAIN_ID.fullmatch(source_id)
        if match:
            numbered.append((int(match.group(1)), row))
    numbered.sort(key=lambda item: item[0])
    return [row for _, row in numbered]


def problem_text(row: dict[str, Any]) -> tuple[str, str]:
    problem = str(
        row.get("translated_question") or row.get("problem") or row.get("question") or ""
    ).strip()
    solution = str(row.get("translated_solution") or row.get("solution") or "").strip()
    return problem, solution


def content_hash(problem: str, solution: str) -> str:
    encoded = json.dumps(
        {"problem": problem, "reference_solution": solution},
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def classify_topic(problem: str, solution: str) -> tuple[str, list[str]]:
    text = f"{problem}\n{solution}".lower()
    scored: list[tuple[int, int, str, list[str]]] = []
    for priority, (topic, patterns) in enumerate(TOPIC_RULES.items()):
        evidence = [pattern for pattern in patterns if re.search(pattern, text)]
        scored.append((
            len(evidence) + (TOPIC_SPECIFICITY_BONUS[topic] if evidence else 0),
            -priority, topic, evidence[:4],
        ))
    score, _, topic, evidence = max(scored)
    return (topic, evidence) if score else ("arithmetic", ["default:no_topic_marker"])


def required_stage(topic: str, problem: str, solution: str) -> int:
    text = f"{problem}\n{solution}".lower()
    stage = BASE_REQUIRED_STAGE[topic]
    advanced_markers = {
        "algebra": [r"quadratic", r"二次方程式", r"discriminant", r"判別式", r"logarith", r"complex"],
        "geometry": [r"similar", r"相似", r"circle theorem", r"円周角", r"sphere", r"球", r"vector", r"ベクトル"],
        "probability_combinatorics": [r"\\binom", r"permutation", r"combination", r"順列", r"組合せ", r"conditional", r"条件付き"],
        "number_theory": [r"modulo", r"合同式", r"diophant", r"不定方程式", r"base[- ]?\d", r"進法"],
        "functions_sequences": [r"logarith", r"対数", r"complex", r"複素数", r"geometric sequence", r"等比数列"],
    }
    if any(re.search(marker, text) for marker in advanced_markers.get(topic, [])):
        stage += 1
    return max(1, min(4, stage))


def scope_relation(required: int, mastery: int, math_level: int) -> str:
    gap = required - mastery
    if gap <= 0:
        relation_index = 0
    elif gap == 1:
        relation_index = 1
    elif gap == 2:
        relation_index = 2
    else:
        relation_index = 3
    if math_level >= 4 and relation_index == 0:
        relation_index = 1
    elif math_level == 5 and relation_index == 1:
        relation_index = 2
    return RELATION_ORDER[relation_index]


def choose_profile(
    profiles: list[dict[str, Any]], topic: str, required: int, math_level: int,
    profile_counts: Counter[str], relation_counts: Counter[str],
) -> tuple[dict[str, Any], str, str]:
    options: list[tuple[int, dict[str, Any], str]] = []
    for order, profile in enumerate(profiles):
        mastery = int(profile["topic_mastery"][topic])
        relation = scope_relation(required, mastery, math_level)
        options.append((order, profile, relation))
    available_relations = {relation for _, _, relation in options}
    target_relation = min(
        available_relations,
        key=lambda relation: (
            relation_counts[relation] / RELATION_TARGET_WEIGHTS[relation],
            RELATION_ORDER.index(relation),
        ),
    )
    matching = [item for item in options if item[2] == target_relation]
    _, profile, relation = min(
        matching,
        key=lambda item: (profile_counts[str(item[1]["id"])], item[0]),
    )
    return profile, relation, target_relation


def initial_emotion(relation: str, math_level: int) -> tuple[str, str]:
    if relation == "mastered":
        if math_level == 1:
            return "neutral", "既習範囲内の定型問題で、強い感情反応を仮定しない"
        if math_level >= 4:
            return "curious", "既習範囲内だが高難度で、方針を探索する余地がある"
        return "engaged", "既習範囲内で自力着手が可能である"
    if relation == "frontier":
        if math_level >= 4:
            return "confused", "現在学習中の範囲かつ高難度で、方針がまだ安定していない"
        return "curious", "現在学習中の範囲で、既習事項との接続を探せる"
    if relation == "one_step_beyond":
        if math_level >= 4:
            return "anxious", "学習範囲を一段超え、難度も高く正答への見通しが弱い"
        return "confused", "学習範囲を一段超え、最初に使う知識を特定できない"
    return "frustrated", "現在の学習範囲から二段階以上離れ、自力解決の見通しがない"


def misconception_for(topic: str, relation: str) -> dict[str, str]:
    if relation in {"one_step_beyond", "far_beyond"}:
        return {
            "id": "M-BOUNDARY-PARTIAL",
            "label": "既習操作だけで着手し、最初の未習概念で停止する",
            "trigger": "問題の必須知識が現在のカリキュラム位置を超える",
            "faulty_procedure": "問題文の量を整理した後、似た既習操作を一度だけ試すが、未習公式を補完しない",
            "observable_signature": "完成解答を出さず、既知条件と具体的な不明点を分離して援助を求める",
            "repair_criterion": "教師が未習概念を明示し、生徒がその一段階を別の数値または式で再現する",
        }
    if relation == "mastered":
        return {
            "id": "M-VERIFY-OMISSION",
            "label": "解法を知っているため条件確認または検算を省略しやすい",
            "trigger": "既習範囲の問題で解法方針がすぐに見つかる",
            "faulty_procedure": "主要計算を進めるが、定義域、場合分け、元の条件への代入の一つを省く",
            "observable_signature": "方針と主要式は妥当だが、最終結論の根拠が一箇所未確認になる",
            "repair_criterion": "省略した条件または検算を自分で補い、結論との整合を説明する",
        }
    return dict(TOPIC_MISCONCEPTIONS[topic])


def prior_attempt_history(emotion: str) -> dict[str, Any]:
    if emotion == "frustrated":
        return {
            "attempt_count": 2,
            "attempted_strategy": "既習範囲の操作を二度試した",
            "repeated_stuck_point": "最初の未習概念が必要になる同じ箇所で停止した",
            "received_help": False,
        }
    return {
        "attempt_count": 0,
        "attempted_strategy": "なし",
        "repeated_stuck_point": "なし",
        "received_help": False,
    }


def build_assignments(
    questions: list[dict[str, Any]], profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    profile_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    assignments: list[dict[str, Any]] = []
    for index, row in enumerate(ordered_questions(questions)):
        source_id = str(row.get("id") or row.get("source_id"))
        problem, solution = problem_text(row)
        topic, evidence = classify_topic(problem, solution)
        stage = required_stage(topic, problem, solution)
        math_level = int(row["level"])
        profile, relation, target_relation = choose_profile(
            profiles, topic, stage, math_level, profile_counts, relation_counts,
        )
        profile_counts[str(profile["id"])] += 1
        relation_counts[relation] += 1
        emotion, emotion_reason = initial_emotion(relation, math_level)
        misconception = misconception_for(topic, relation)
        attempt_history = prior_attempt_history(emotion)
        if relation == "mastered":
            response_mode = "partial_reasoning" if index % 2 == 0 else "correct_but_uncertain"
        elif relation == "frontier":
            response_mode = "plausible_incorrect" if index % 2 == 0 else "partial_reasoning"
        else:
            response_mode = "scope_limited_help_seeking"
        assignments.append({
            "source_id": source_id,
            "order_index": index,
            "question_sha256": content_hash(problem, solution),
            "policy_version": POLICY_VERSION,
            "curriculum_annotation": {
                "topic": topic,
                "required_stage": stage,
                "required_scope": SCOPE_LABELS[stage],
                "rule_evidence": evidence,
                "math_level": math_level,
                "annotation_method": "deterministic_rules",
                "requires_human_review": evidence == ["default:no_topic_marker"],
            },
            "profile_id": profile["id"],
            "profile_topic_mastery": int(profile["topic_mastery"][topic]),
            "scope_relation": relation,
            "target_scope_relation": target_relation,
            "initial_emotion": emotion,
            "initial_emotion_reason": emotion_reason,
            "prior_attempt_history": attempt_history,
            "misconception_model": misconception,
            "initial_response_mode": response_mode,
            "initial_response_constraint": (
                "scope_limited_help_seeking"
                if relation in {"one_step_beyond", "far_beyond"}
                else "profile_consistent_attempt"
            ),
        })
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
    assignments = build_assignments(read_jsonl(args.questions), profiles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in assignments:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter(row["profile_id"] for row in assignments)
    relations = Counter(row["scope_relation"] for row in assignments)
    emotions = Counter(row["initial_emotion"] for row in assignments)
    print(json.dumps({
        "output": str(args.output), "assignments": len(assignments),
        "profile_counts": counts, "scope_relation_counts": relations,
        "initial_emotion_counts": emotions,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
