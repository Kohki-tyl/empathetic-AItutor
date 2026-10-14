"""Package the existing Japanese controls without changing their fixed splits."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v4'))
import train_v4_sft as engine

ROOT = Path(__file__).resolve().parents[3]
DESTINATION = ROOT / 'research/corpus/sft_1000'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def package(condition, rows, provenance):
    engine.validate_messages(rows, len(rows), 'plain')
    splits = Counter(row['split'] for row in rows)
    if len(rows) != 1000 or splits != {'train': 900, 'validation': 100}:
        raise ValueError(f'{condition}: expected exactly 900 train and 100 validation')
    groups = {}
    for row in rows:
        group = row['group_id']
        if group in groups and groups[group] != row['split']:
            raise ValueError(f'Group leakage: {group}')
        groups[group] = row['split']
        for message in row['messages']:
            if any(ord(c) == 127 or (ord(c) < 32 and c not in '\r\n\t') for c in message['content']):
                raise ValueError(f'Control character: {row["id"]}')
    destination = DESTINATION / condition
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    write_rows(destination / 'data.jsonl', rows)
    for split in ('train', 'validation'):
        write_rows(destination / f'{split}.jsonl', [r for r in rows if r['split'] == split])
    config = json.loads((ROOT / 'pipelines/sft/v4/config.json').read_text(encoding='utf-8'))
    config.update(dataset='data.jsonl', dataset_sha256=engine.sha256_file(destination / 'data.jsonl'),
                  expected_records=1000, response_format='plain', analysis_loss_weight=1.0,
                  final_loss_weight=1.0, output_dir=f'outputs/{condition}_ja1000', seed=42)
    write_json(destination / 'config.json', config)
    manifest = dict(condition=condition, records=len(rows), split_counts=dict(splits),
                    assistant_turns=sum(m['role'] == 'assistant' for r in rows for m in r['messages']),
                    format='plain assistant-only loss; teacher content only', provenance=provenance,
                    files={name: engine.sha256_file(destination / name) for name in ('data.jsonl', 'train.jsonl', 'validation.jsonl', 'config.json')},
                    tokenizer_preflight='not yet run; training refuses overlength dialogues',
                    train_ids=[r['id'] for r in rows if r['split'] == 'train'],
                    validation_ids=[r['id'] for r in rows if r['split'] == 'validation'])
    write_json(destination / 'manifest.json', manifest)
    (destination / 'train.py').write_text('''"""Run the shared SFT trainer with this folder's fixed configuration."""
from pathlib import Path
import runpy
import sys

folder = Path(__file__).resolve().parent
if '--config' in sys.argv[1:]:
    raise SystemExit('This entry point uses its own config.json')
sys.argv[1:1] = ['--config', str(folder / 'config.json')]
runpy.run_path(str(folder.parents[3] / 'pipelines/sft/v4/train_v4_sft.py'), run_name='__main__')
''', encoding='utf-8')
    (destination / 'README.md').write_text(f'''# {condition} 日本語SFT

全1,000対話（学習900・検証100）。`data.jsonl`に明示splitを持ち、学習処理はこの区分を維持する。
`train.jsonl`と`validation.jsonl`は同じ内容の分割コピー。データの追加生成や再翻訳はしていない。
発話本文を通常応答形式で学習し、lossはassistantのみに適用する。3条件の比較では教師の内部推論を学習対象に含めない。

リポジトリルートから実行する（共通の `pipelines/sft/v4` も必要）。

```bash
python research/corpus/sft_1000/{condition}/train.py --structure-only
python research/corpus/sft_1000/{condition}/train.py --preflight-only
python research/corpus/sft_1000/{condition}/train.py --resume auto
```

依存関係は `pipelines/sft/v4/requirements.txt`。基盤モデル・revision・LoRA等は既存v4設定を引き継ぐ。
実学習はCUDA/BF16対応GPUが必要。tokenizerでの長さ検査はpreflightで行い、長すぎる対話は暗黙に切り詰めず停止する。
1,000件は学習・検証の合計。対話数をそろえても教師token数は一致しない。
来歴・入力hash・採用IDは `manifest.json` を参照。機械翻訳の意味の忠実性は全件の人手レビュー前。
''', encoding='utf-8')
    print(f'{condition}: {dict(splits)}, {manifest["assistant_turns"]} assistant turns')


def main():
    for condition in ('math', 'empathy'):
        source = ROOT / f'research/corpus/baselines_ja1000/{condition}_ja_1000.jsonl'
        rows = engine.read_jsonl(source)
        package(condition, rows, dict(source=source.relative_to(ROOT).as_posix(),
                                      sha256=engine.sha256_file(source),
                                      translation='GPT-6 Luna; existing Japanese translation preserved',
                                      selection='Existing 1000 dialogues and fixed 900/100 splits preserved'))


if __name__ == '__main__':
    main()
