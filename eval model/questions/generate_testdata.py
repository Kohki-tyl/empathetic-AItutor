import json
import os

def create_test_dataset(input_filename, output_filename, num_samples=200):
    if not os.path.exists(input_filename):
        print(f"エラー: 入力ファイル '{input_filename}' が見つかりません。")
        return

    # すべての行（データ）を読み込む
    with open(input_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"入力ファイルの総データ数: {total_lines}件")

    # データ数が指定されたサンプル数より少ない場合は、全データを対象にする
    if total_lines < num_samples:
        print(f"警告: データ数が{num_samples}件に満たないため、全データを抽出します。")
        test_lines = lines
    else:
        # 後ろから指定件数をスライスして取得
        test_lines = lines[-num_samples:]

    # 出力ファイルに書き込む
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)

    print(f"抽出完了: 後ろから {len(test_lines)} 件のデータを '{output_filename}' に保存しました。")

if __name__ == "__main__":
    import os
    
    # このPythonスクリプト自身が保存されているディレクトリの絶対パスを取得
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # スクリプトと同じ階層にある前提で、ファイルの絶対パスを結合
    INPUT_FILE = os.path.join(SCRIPT_DIR, "translated_1000_math.jsonl")
    OUTPUT_FILE = os.path.join(SCRIPT_DIR, "test_math_questions.jsonl")
    
    print(f"読み込み先パス: {INPUT_FILE}") # デバッグ用
    
    create_test_dataset(INPUT_FILE, OUTPUT_FILE)