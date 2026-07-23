import json
from pathlib import Path

# ==========================================
# 1. 基本設定
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

def load_results(filename: str) -> dict:
    data = {}
    filepath = BASE_DIR / filename
    if not filepath.exists():
        return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data[item["source_id"]] = item
    return data

# ==========================================
# 2. メインの分析処理
# ==========================================
def analyze():
    print("データを読み込んで分析を開始します...\n")
    
    # ★ FTモデルの結果ファイルのみを読み込む
    ft_data = load_results("baseline_evaluated_results.jsonl")

    if not ft_data:
        print("エラー: ファイル が見つからないか、データが空です。")
        return

    total_questions = len(ft_data)

    print(f"{'='*40}")
    print(f" 分析結果レポート (FTモデル単独 / 対象: {total_questions}問)")
    print(f"{'='*40}")

    # 変数の初期化
    one_turn_count = 0
    valid_correct_count = 0
    valid_total_count = 0
    
    emotion_sum = 0
    pedagogy_sum = 0
    length_sum = 0
    total_sum = 0
    valid_empathy_count = 0

    # データの集計
    for q_id, item in ft_data.items():
        
        # [指標1 & 2用] ターン数の確認 (1ターン以下をカウント)
        turns = item.get("phase1_turns", 0)
        if turns <= 1:
            one_turn_count += 1
        else:
            valid_total_count += 1
            if item.get("phase2_is_correct"):
                valid_correct_count += 1
                
        # [指標3用] 共感スコアの集計
        emp = item.get("empathy_evaluation")
        if emp:
            emotion_sum += emp.get("emotion_alignment_score", 0)
            pedagogy_sum += emp.get("pedagogical_empathy_score", 0)
            length_sum += emp.get("length_control_score", 0)
            total_sum += emp.get("total_score", 0)
            valid_empathy_count += 1

    # --------------------------------------------------
    # 出力1: Phase1の1ターンで終了している割合
    # --------------------------------------------------
    one_turn_ratio = (one_turn_count / total_questions) * 100
    print(f"\n▼ [指標1] 対話放棄（Phase1が1ターン以内で終了）の割合")
    print(f"該当数: {one_turn_count}問 / {total_questions}問 ({one_turn_ratio:.1f}%)")
    if one_turn_ratio > 30:
        print("  ⚠️ 注意: 1ターン終了が多すぎます。モデルが「対話を通じた指導」をすぐに諦めている可能性があります。")

    # --------------------------------------------------
    # 出力2: 1ターン終了を除外した実質的な転移テスト正答率
    # --------------------------------------------------
    print(f"\n▼ [指標2] 実質的な転移テスト正答率 (1ターン終了の {one_turn_count}問 を除外)")
    if valid_total_count > 0:
        acc = (valid_correct_count / valid_total_count) * 100
        print(f"有効対話数: {valid_total_count}問")
        print(f"正答率    : {acc:.1f}% ({valid_correct_count}問正解)")
    else:
        print("  有効な対話（2ターン以上）が行われたセッションがありませんでした。")

    # --------------------------------------------------
    # 出力3: スコアリングの各項目の平均
    # --------------------------------------------------
    print(f"\n▼ [指標3] 共感スコアリング各項目の平均 (全{total_questions}問)")
    if valid_empathy_count > 0:
        print(f"感情の寄り添い (Emotion Alignment) : {emotion_sum / valid_empathy_count:.1f} 点")
        print(f"教育的配慮 (Pedagogical Empathy)   : {pedagogy_sum / valid_empathy_count:.1f} 点")
        print(f"長さの制御 (Length Control)        : {length_sum / valid_empathy_count:.1f} 点")
        print(f"----------------------------------------")
        print(f"合計スコア平均 (Total Score)       : {total_sum / valid_empathy_count:.1f} 点")
    else:
        print("  共感スコアのデータが見つかりませんでした。")
        
    print("\n")

if __name__ == "__main__":
    analyze()