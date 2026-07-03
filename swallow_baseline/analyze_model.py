import json
import statistics
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def main():
    print("ベースライン評価結果の分析を開始します...\n")
    
    # 読み込むファイル名（先ほどアップロードいただいたファイル）
    input_file = BASE_DIR / "evaluation_swallow_baseline_results.jsonl"
    
    if not input_file.exists():
        print(f"[エラー] ファイルが見つかりません: {input_file}")
        return

    total_sessions = 0
    phase1_completed_count = 0
    phase2_correct_count = 0
    
    # スコア集計用リスト
    empathy_totals = []
    emotion_alignments = []
    pedagogical_empathies = []
    length_controls = []
    turn_counts = []

    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            total_sessions += 1
            
            # 1. 成功率のカウント
            if data.get("phase1_is_completed", False):
                phase1_completed_count += 1
            if data.get("phase2_is_correct", False):
                phase2_correct_count += 1
                
            # 2. ターン数の記録
            turn_counts.append(data.get("phase1_turns", 0))
            
            # 3. 共感スコアの記録
            emp_eval = data.get("empathy_evaluation", {})
            empathy_totals.append(emp_eval.get("total_score", 0))
            emotion_alignments.append(emp_eval.get("emotion_alignment_score", 0))
            pedagogical_empathies.append(emp_eval.get("pedagogical_empathy_score", 0))
            length_controls.append(emp_eval.get("length_control_score", 0))

    if total_sessions == 0:
        print("データが空です。")
        return

    # --- 結果の計算と出力 ---
    print("="*40)
    print("📊 ベースライン評価サマリー (Swallow-70B)")
    print("="*40)
    print(f"総評価セッション数 : {total_sessions}件")
    print(f"Phase 1 指導完了率 : {phase1_completed_count / total_sessions * 100:.1f}% ({phase1_completed_count}/{total_sessions})")
    print(f"Phase 2 テスト正答率: {phase2_correct_count / total_sessions * 100:.1f}% ({phase2_correct_count}/{total_sessions})")
    print(f"平均対話ターン数   : {statistics.mean(turn_counts):.1f} ターン")
    
    print("\n" + "="*40)
    print("💖 共感・足場かけスコア (平均値)")
    print("="*40)
    print(f"総合スコア (30点満点)       : {statistics.mean(empathy_totals):.2f} 点")
    print("-" * 40)
    print(f"感情アライメント (10点満点) : {statistics.mean(emotion_alignments):.2f} 点")
    print(f"教育的共感/足場 (10点満点)  : {statistics.mean(pedagogical_empathies):.2f} 点")
    print(f"発話の長さ制御 (10点満点)   : {statistics.mean(length_controls):.2f} 点")
    print("="*40)

    # 分析結果をテキストファイルにも保存
    output_txt = BASE_DIR / "baseline_summary_report.txt"
    with output_txt.open("w", encoding="utf-8") as f:
        f.write("ベースライン評価サマリー (Swallow-70B)\n")
        f.write(f"総セッション数: {total_sessions}\n")
        f.write(f"指導完了率: {phase1_completed_count / total_sessions * 100:.1f}%\n")
        f.write(f"テスト正答率: {phase2_correct_count / total_sessions * 100:.1f}%\n")
        f.write(f"総合スコア平均: {statistics.mean(empathy_totals):.2f}\n")
    
    print(f"\n[INFO] 概要レポートを {output_txt.name} に保存しました。SFT後の比較にお使いください。")

if __name__ == "__main__":
    main()