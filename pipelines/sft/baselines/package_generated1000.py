"""Select 1000 keep dialogues under the user-approved relaxed surface policy."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

from package_ja1000 import ROOT, DESTINATION, package, write_json
from prepare import SYSTEM

sys.path.insert(0, str(ROOT / 'pipelines/corpus_creation'))
from continue_corpus import repair_surface

CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def load(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / 'research/corpus/empathetic_dialogues_1300/1300_empathetic_dialogues.jsonl'
    audit_paths = [ROOT / 'pipelines/corpus_creation/800_dialogue_evaluations.jsonl',
                   ROOT / 'research/corpus/empathetic_dialogues_1300/audit/additional_500_dialogue_evaluations_801_1300.jsonl']
    audits = {r['source_id']: r for p in audit_paths for r in load(p)}
    corpus = load(source)
    assert len(corpus) == len(audits) == 1300
    candidates = []
    for row in corpus:
        a = audits[row['source_id']]
        e = a['evaluation']
        if (a.get('status') != 'completed' or not row['is_completed'] or
                e['recommendation'] != 'keep' or e['critical_failure'] or e['false_affirmation_count']):
            continue
        all_strings = [v for t in row['conversation'] for v in t.values() if isinstance(v, str)]
        strict_surface = not any(CONTROL.search(v) for v in all_strings)
        content_controls = sum(len(CONTROL.findall(t['content'])) for t in row['conversation'])
        candidates.append((row, e, strict_surface, content_controls))
    # Preserve all 772 previously eligible IDs, then prefer fewer content defects and higher scores.
    candidates.sort(key=lambda x: (not x[2], x[3], -x[1]['total_score'],
                                  hashlib.sha256(('42:' + x[0]['source_id']).encode()).hexdigest()))
    assert len(candidates) >= 1000
    selected = candidates[:1000]
    source_ids = [x[0]['source_id'] for x in selected]
    assert len(set(source_ids)) == 1000
    shuffled = sorted(source_ids)
    random.Random(42).shuffle(shuffled)
    validation = set(shuffled[:100])
    rows, changes, quality = [], [], []
    for raw, evaluation, strict_surface, content_controls in selected:
        sid = raw['source_id']
        messages = [dict(role='system', content=SYSTEM)]
        for index, turn in enumerate(raw['conversation']):
            original = turn['content']
            # Only transform strings containing prohibited controls; preserve clean strings verbatim.
            text = CONTROL.sub('', repair_surface(original)) if CONTROL.search(original) else original
            if not text.strip():
                raise ValueError(f'Empty normalized content: {sid}:{index}')
            if text != original:
                changes.append(dict(source_id=sid, conversation_index=index, role=turn['role'],
                                    turn=turn['turn'], original=original, normalized=text,
                                    requires_semantic_review=True))
            if index == 0:
                problem = raw['problem']
                if CONTROL.search(problem):
                    raise ValueError('Unexpected control character in problem')
                text = problem + '\n\n' + text
            messages.append(dict(role='user' if turn['role'] == 'student' else 'assistant', content=text))
        rows.append(dict(id='eduempathy:' + sid, source='generated_empathetic_math', group_id=sid,
                         language='ja', split='validation' if sid in validation else 'train', messages=messages))
        quality.append(dict(source_id=sid, recommendation=evaluation['recommendation'],
                            total_score=evaluation['total_score'], strict_previous_policy_passed=strict_surface,
                            source_content_control_count=content_controls,
                            surface_normalization_applied=any(c['source_id'] == sid for c in changes),
                            semantic_reaudit_after_normalization=False))
    random.Random(42).shuffle(rows)
    package('eduempathy', rows, dict(source=source.relative_to(ROOT).as_posix(), sha256=sha(source),
        audits={p.relative_to(ROOT).as_posix(): sha(p) for p in audit_paths}, seed=42,
        eligible_content_pool=len(candidates), selection='All 772 strict candidates, then fewer content control characters, higher audit score, seeded hash tie-break',
        strict_candidates_included=sum(x[2] for x in selected),
        relaxed_surface_candidates=sum(not x[2] for x in selected),
        authorization='User explicitly chose relaxed quality conditions on 2026-10-15',
        policy='Keep, completed, no critical failure, no false affirmation; surface defects allowed with logged normalization; no semantic re-audit',
        normalization='Existing repair_surface for known escapes, followed by removal of remaining C0/DEL; only on content containing prohibited controls',
        limitations='Normalization may affect mathematical meaning; candidate status does not guarantee correctness. Original data unchanged.'))
    output = DESTINATION / 'eduempathy'
    write_json(output / 'surface_changes.json', changes)
    write_json(output / 'quality_metadata.json', quality)
    with (output / 'README.md').open('a', encoding='utf-8') as handle:
        handle.write('''
## 生成対話の選定条件

元の1,300対話から、内容監査keep・完了・重大失敗なし・誤答肯定0の1,106対話を候補とした。
前回の厳格な採用候補772件を全件含め、表記条件を緩めた228件を追加して1,000件にした。
要確認・rejectという内容判定の対話は採用していない。教師の内部フィールドは学習入力から除外する。
初回user入力には問題文と最初の生徒発話をまとめて含める。

制御文字のある本文は既知のLaTeXエスケープを復元したうえで残存C0/DELを除去した。
修正前後は`surface_changes.json`、元の品質・変更有無は`quality_metadata.json`に記録した。
これは形式の正規化であり、数式の意味や元表記の復元を保証しない。正規化後の内容再監査は未実施。
生成対話の元データと監査結果は変更していない。翻訳の注意書きは元問題に関するもので、対話本文は生成データ。
''')
    print('Selected quality:', dict(Counter('strict' if x[2] else 'relaxed' for x in selected)))
    print('Normalized fields:', len(changes), 'dialogues:', len({c['source_id'] for c in changes}))


if __name__ == '__main__':
    main()
