import json
import sys
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def load_system_prompt() -> str:
    path = BASE_DIR / "prompts" / "sft_teacher_system.txt"
    if not path.exists():
        # プロンプトが読み込めない場合はエラーを出力して強制終了
        print(f"\n[Error] システムプロンプトファイルが見つかりません！\nパスを確認してください: {path}", file=sys.stderr)
        sys.exit(1)
        
    return path.read_text(encoding="utf-8").strip()

def main():
    print("コーパス (empathetic_dialogues.jsonl) から学習用・検証用データを作成します...")
    
    system_prompt = load_system_prompt()
    input_path = BASE_DIR / "empathetic_dialogues.jsonl"
    train_output_path = BASE_DIR / "sft_train.jsonl"
    val_output_path = BASE_DIR / "sft_val.jsonl"
    
    if not input_path.exists():
        print(f"\n[Error] 入力ファイルが見つかりません: {input_path}", file=sys.stderr)
        sys.exit(1)

    total_count = 0
    valid_data = [] # 条件をクリアしたデータを一時保存するリスト
    
    with input_path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            if not line.strip(): continue
            total_count += 1
            session = json.loads(line)
            
            # フィルタリング: 指導完了フラグのみで判定
            if not session.get("is_completed", False): 
                continue

            problem_text = session.get("problem", "")
            conversation = session.get("conversation", [])
            
            # messagesフォーマットの初期化（Systemプロンプトを設定）
            messages = [{"role": "system", "content": system_prompt}]
            
            for i, turn in enumerate(conversation):
                role = "user" if turn["role"] == "student" else "assistant"
                content = turn.get("content", "").strip()
                
                # 初回ターンのユーザー発話に問題文とシステム指示を結合
                if role == "user" and i == 0:
                    content = f"問題: {problem_text}\n\n上記の問題を出題しました。生徒の発話を待機し、対応を開始してください。\n\n{content}"
                
                # 最終ターンのアシスタント発話に [指導完了] タグを付与
                elif role == "assistant":
                    is_last_turn = (i == len(conversation) - 1)
                    if is_last_turn and "[指導完了]" not in content:
                        content += "\n\n[指導完了]"
                        
                messages.append({"role": role, "content": content})
            
            valid_data.append({"messages": messages})
            
    # ==========================================
    # データのシャッフルと 8:2 分割処理
    # ==========================================
    valid_count = len(valid_data)
    if valid_count == 0:
        print("\n[Error] 有効なデータが1件もありませんでした。", file=sys.stderr)
        sys.exit(1)

    # ランダムシードを固定して、何度実行しても同じ分割結果になるようにする
    random.seed(42)
    random.shuffle(valid_data)
    
    # 8割のインデックスを計算
    split_idx = int(valid_count * 0.8)
    
    train_data = valid_data[:split_idx]
    val_data = valid_data[split_idx:]
    
    # トレーニング用データの書き込み (80%)
    with train_output_path.open("w", encoding="utf-8") as f_out:
        for item in train_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # 検証用データの書き込み (20%)
    with val_output_path.open("w", encoding="utf-8") as f_out:
        for item in val_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\n✅ 変換・分割完了!")
    print(f" - 全データ数: {total_count} 件")
    print(f" - 抽出された高品質データ: {valid_count} 件")
    print(f"   => トレーニング用 (80%): {len(train_data)} 件 -> {train_output_path.name}")
    print(f"   => 検証用 (20%): {len(val_data)} 件 -> {val_output_path.name}")

if __name__ == "__main__":
    main()