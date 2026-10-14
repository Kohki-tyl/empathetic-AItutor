"""Download official source data, normalize and randomly sample complete dialogues."""
import argparse
import csv
import io
import json
import random
import tarfile
import urllib.request
from pathlib import Path

import prepare as p


def download(url, path):
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "empathetic-AItutor-research"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    return path


def sample(rows, size, seed):
    if size > len(rows):
        raise ValueError(f"Requested {size} dialogues, only {len(rows)} available")
    return random.Random(seed).sample(sorted(rows, key=lambda r: r["id"]), size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("sample-size must be positive")
    root = args.output
    if (root / "source_manifest.json").exists():
        raise FileExistsError("Source preparation is already complete; use existing samples or a new output")
    commit_path = download("https://api.github.com/repos/eth-nlped/mathdial/commits/main", root / "raw/mathdial_commit.json")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))["sha"]
    math_url = f"https://raw.githubusercontent.com/eth-nlped/mathdial/{commit}/data/train.jsonl"
    math_path = download(math_url, root / "raw/mathdial_train.jsonl")
    math, rejected = [], []
    for index, raw in enumerate(p.engine.read_jsonl(math_path)):
        try:
            row = p.mathdial([raw], args.seed)[0]
            row["id"] = f"mathdial:{raw['qid']}:{index}"
            math.append(row)
        except ValueError as exc:
            rejected.append({"raw_index": index, "qid": str(raw["qid"]), "reason": str(exc)})
    ed_url = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
    archive = download(ed_url, root / "raw/empatheticdialogues.tar.gz")
    empathy, rejected_ed = [], []
    # Read only exact regular-file members; never extract archive paths onto disk.
    with tarfile.open(archive, "r:gz") as bundle:
        for name, split in (("train.csv", "train"), ("valid.csv", "validation")):
            member = bundle.getmember(f"empatheticdialogues/{name}")
            if not member.isfile():
                raise ValueError(f"Unexpected archive member: {name}")
            with bundle.extractfile(member) as binary:
                with io.TextIOWrapper(binary, encoding="utf-8", newline="") as stream:
                    groups = {}
                    for turn in csv.DictReader(stream):
                        groups.setdefault(turn["conv_id"], []).append(turn)
                    for cid, turns in groups.items():
                        try:
                            empathy.extend(p.empathetic(turns, split))
                        except ValueError as exc:
                            rejected_ed.append({"conv_id": cid, "split": split, "reason": str(exc)})
    manifest = {"seed": args.seed, "sample_size": args.sample_size, "mathdial_commit": commit,
                "sources": {math_url: p.engine.sha256_file(math_path), ed_url: p.engine.sha256_file(archive)},
                "datasets": {}, "rejected_mathdial": rejected, "rejected_empathy": rejected_ed}
    for name, rows in (("math", math), ("empathy", empathy)):
        selected = sample(rows, args.sample_size, args.seed)
        p.engine.validate_messages(selected, len(selected), "plain")
        p.engine.deterministic_split(selected, 0.1, args.seed)
        path = root / f"{name}_en_{args.sample_size}.jsonl"
        p.write_rows(path, selected)
        manifest["datasets"][name] = {"available_records": len(rows), "sample_sha256": p.engine.sha256_file(path),
                                         "selected_ids": [r["id"] for r in selected],
                                         "train": sum(r["split"] == "train" for r in selected),
                                         "validation": sum(r["split"] == "validation" for r in selected)}
        print(f"{name}: sampled {len(selected)} of {len(rows)} dialogues -> {path}")
    p.engine.write_json(root / "source_manifest.json", manifest)


if __name__ == "__main__":
    main()
