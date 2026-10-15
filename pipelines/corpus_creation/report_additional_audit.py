"""Combine the completed content audit with local formatting exclusions."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / 'research/corpus/empathetic_dialogues_1300'


def main():
    out = BASE / 'audit'
    rows = [json.loads(line) for line in (out / 'additional_500_dialogue_evaluations_801_1300.jsonl').read_text(encoding='utf-8').splitlines()]
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    local = json.loads((out / 'local_audit.json').read_text(encoding='utf-8'))
    assert len(rows) == len({r['source_id'] for r in rows}) == 500
    problematic = {r[0] for r in local['unsupported_control_characters']}
    content_candidates = {r['source_id'] for r in rows if r['sft_eligible']}
    candidates = content_candidates - problematic
    summary['content_candidates_with_control_characters'] = len(content_candidates & problematic)
    summary['sft_candidates_after_local_checks'] = len(candidates)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    selection = json.loads((out / 'selection.json').read_text(encoding='utf-8'))
    selection['sft_candidate_ids'] = [r['source_id'] for r in rows if r['source_id'] in candidates]
    selection['content_candidate_ids_before_local_checks'] = [r['source_id'] for r in rows if r['sft_eligible']]
    selection['format_review_ids'] = sorted(problematic)
    selection['excluded_from_sft_ids'] = [r['source_id'] for r in rows if r['source_id'] not in candidates]
    (out / 'selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
    local['audit_status'] = 'local_and_content_checks_completed'
    local['content_quality_audited'] = True
    local['sft_selection_finalized'] = False
    local['selection_note'] = 'Candidate IDs available in selection.json; no training data or split changed.'
    (out / 'local_audit.json').write_text(json.dumps(local, ensure_ascii=False, indent=2), encoding='utf-8')
    text = '''# 追加500対話の監査結果

2026-10-14実施。500件全件の内容監査とローカル検査を完了した。元データは変更していない。

## 内容監査

| 判定 | 対話数 |
|---|---:|
'''
    for key in ('keep', 'review', 'reject'):
        text += f"| {key} | {summary['recommendations'].get(key, 0)} |\n"
    text += f"| 重大な問題あり（上記判定との重複あり） | {summary['critical_failure_dialogues']} |\n"
    text += f"| 誤答の肯定あり（同上） | {summary['false_affirmation_dialogues']} |\n"
    text += f"\n平均合計点は {summary['averages']['total_score']:.2f}/60。誤答の肯定は合計{summary['false_affirmation_events']}回。\n"
    text += f'''
## SFT候補とローカル検査

内容判定keep・完了申告true・重大な問題なし・誤答肯定0回をすべて満たす候補は{len(content_candidates)}件。
うち制御文字が残る{len(content_candidates & problematic)}件を除外すると、SFT採用候補は**{len(candidates)}件**。
これは候補の抽出であり、学習ファイルの作成やtrain/validation分割の変更は行っていない。

全体では完了申告484件、未完了16件。29対話・44フィールドにDEL制御文字U+007Fが151文字残っている。
数式の区切りやノルム表記に関わるため単純削除だけでの修復は保証できない。
ID・問題文の完全一致重複0、参照問題との対応不一致0、既存の構造検証エラー0。
既存800対話と結合版の先頭は一致し、追加500対話と結合版の後半も一致した。

## 方法・限界

モデルは `{', '.join(summary['resolved_models'])}`、temperature 0、reasoning effort none。
感情整合・教育的共感・数学的正確性・誤り検出と回復・適応的足場かけ・長さ制御の6軸を各0〜10点で評価した。
対話全体（教師の内部フィールドも含む）と原文問題・英語参照解答・日本語訳を提示し、参照解答自体も疑って検算するよう指定した。
旧800件と6軸・スキーマは共通だが、参照解答を加えたため入力条件は同一ではない。
教師生成と同系列の単一LLMによる判定であり、人手検証や数学的証明による正しさの保証ではない。
制御文字を含む原文も変更せず評価し、ローカル検査で別途除外した。

最初のAPI実行は自動承認レビューで拒否されたが、その後ユーザーの明示承認を受けて全件実行した。
再開可能なチェックポイントを保存し、全500件のID一意性・判定形式・点数合計・入力ハッシュを確認した。

## ファイル

- [全500件の評価・具体的理由](additional_500_dialogue_evaluations_801_1300.jsonl)
- [集計](summary.json)
- [採用候補・要確認・除外ID](selection.json)
- [ローカル検査詳細](local_audit.json)
- [モデル・プロンプト・入力ハッシュ](manifest.json)
'''
    text += '\n## 要確認例\n\n'
    for row in [r for r in rows if r['evaluation']['critical_failure']][:5]:
        text += f"### {row['source_id']}\n\n{row['evaluation']['reason']}\n\n"
    (out / 'README.md').write_text(text, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
