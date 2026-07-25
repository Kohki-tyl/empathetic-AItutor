"""v4厳格監査コーパスをBF16 LoRAでSFTする。

学習前に入力構造、系列長、assistant-only lossの位置を全件監査する。
自動切り詰めは行わず、実行条件とsplitをmanifestへ固定する。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Protocol, Sequence


REQUIRED_CONFIG = {
    "dataset",
    "expected_records",
    "model_name",
    "model_revision",
    "output_dir",
    "max_length",
    "validation_ratio",
    "seed",
    "num_train_epochs",
    "learning_rate",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "warmup_ratio",
    "weight_decay",
    "max_grad_norm",
    "lr_scheduler_type",
    "optim",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "gradient_checkpointing",
    "bf16",
    "tf32",
    "dataloader_num_workers",
    "save_total_limit",
    "logging_steps",
}


class ChatTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any: ...


@dataclass(frozen=True)
class EncodedRecord:
    record_id: str
    input_ids: list[int]
    labels: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto、none、またはcheckpointディレクトリ",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON objectではありません: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}がJSON objectではありません")
            rows.append(value)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def resolve_config(path: Path) -> tuple[dict[str, Any], Path, Path]:
    path = path.resolve()
    config = read_json(path)
    missing = sorted(REQUIRED_CONFIG - config.keys())
    if missing:
        raise ValueError(f"configの必須項目がありません: {missing}")

    positive_ints = (
        "expected_records",
        "max_length",
        "seed",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "lora_r",
        "lora_alpha",
        "dataloader_num_workers",
        "save_total_limit",
        "logging_steps",
    )
    for name in positive_ints:
        minimum = 0 if name == "dataloader_num_workers" else 1
        if not isinstance(config[name], int) or config[name] < minimum:
            raise ValueError(f"{name}は{minimum}以上の整数にしてください")
    if not 0 < float(config["validation_ratio"]) < 0.5:
        raise ValueError("validation_ratioは0より大きく0.5未満にしてください")
    if float(config["num_train_epochs"]) <= 0:
        raise ValueError("num_train_epochsは正にしてください")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rateは正にしてください")
    if not 0 <= float(config["warmup_ratio"]) < 1:
        raise ValueError("warmup_ratioは0以上1未満にしてください")
    if not 0 <= float(config["lora_dropout"]) < 1:
        raise ValueError("lora_dropoutは0以上1未満にしてください")
    if not config["model_revision"]:
        raise ValueError("再現性のためmodel_revisionを固定してください")
    modules = config["lora_target_modules"]
    if not isinstance(modules, list) or not modules or not all(
        isinstance(item, str) and item for item in modules
    ):
        raise ValueError("lora_target_modulesは空でない文字列配列にしてください")

    dataset = (path.parent / str(config["dataset"])).resolve()
    output_dir = (path.parent / str(config["output_dir"])).resolve()
    return config, dataset, output_dir


def validate_messages(rows: list[dict[str, Any]], expected_records: int) -> None:
    if len(rows) != expected_records:
        raise ValueError(
            f"レコード数が想定と異なります: expected={expected_records}, actual={len(rows)}"
        )
    seen: set[str] = set()
    for index, row in enumerate(rows):
        record_id = str(row.get("id", "")).strip()
        if not record_id or record_id in seen:
            raise ValueError(f"欠損または重複したid: record={index}, id={record_id!r}")
        seen.add(record_id)
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError(f"messagesが不正です: {record_id}")
        if not all(isinstance(message, dict) for message in messages):
            raise ValueError(f"messageはJSON objectである必要があります: {record_id}")
        roles = [message.get("role") for message in messages]
        expected_roles = ["system"] + [
            "user" if position % 2 == 0 else "assistant"
            for position in range(len(messages) - 1)
        ]
        if roles != expected_roles or roles[-1] != "assistant":
            raise ValueError(
                f"role順が不正です: {record_id}: actual={roles}, expected={expected_roles}"
            )
        for message_index, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"空のmessage: {record_id}:{message_index}")
            if message["role"] == "assistant":
                markers = ("<analysis>", "</analysis>", "<final>", "</final>")
                match = re.fullmatch(
                    r"<analysis>\s*(.*?)\s*</analysis>\s*"
                    r"<final>\s*(.*?)\s*</final>",
                    content.strip(),
                    flags=re.DOTALL,
                )
                if (
                    any(content.count(marker) != 1 for marker in markers)
                    or match is None
                    or not all(part.strip() for part in match.groups())
                ):
                    raise ValueError(
                        f"analysis/final形式が不正です: {record_id}:{message_index}"
                    )


def deterministic_split(
    rows: list[dict[str, Any]], validation_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 2:
        raise ValueError("学習・検証分割には2件以上必要です")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    validation_count = max(1, int(round(len(rows) * validation_ratio)))
    validation_indices = set(order[:validation_count])
    train = [row for index, row in enumerate(rows) if index not in validation_indices]
    validation = [row for index, row in enumerate(rows) if index in validation_indices]
    if not train or not validation:
        raise ValueError("学習または検証集合が空です")
    return train, validation


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise TypeError("chat templateがtoken ID配列を返しませんでした")
    return value


def render_ids(
    tokenizer: ChatTokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    return _token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    )


def encode_with_assistant_mask(
    tokenizer: ChatTokenizer,
    row: dict[str, Any],
    max_length: int,
) -> EncodedRecord:
    record_id = str(row["id"])
    messages = row["messages"]
    full_ids = render_ids(tokenizer, messages, add_generation_prompt=False)
    if len(full_ids) > max_length:
        raise ValueError(
            f"系列長超過（自動切り詰め禁止）: {record_id}: "
            f"tokens={len(full_ids)}, max_length={max_length}"
        )
    labels = [-100] * len(full_ids)
    assistant_messages = 0
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        assistant_messages += 1
        before = render_ids(
            tokenizer,
            messages[:index],
            add_generation_prompt=True,
        )
        after = render_ids(
            tokenizer,
            messages[: index + 1],
            add_generation_prompt=False,
        )
        if len(after) <= len(before) or after[: len(before)] != before:
            raise ValueError(
                f"assistant境界をchat templateから一意に決定できません: "
                f"{record_id}:{index}"
            )
        if len(after) > len(full_ids) or full_ids[: len(after)] != after:
            raise ValueError(
                f"chat templateのprefix性が成立しません: {record_id}:{index}"
            )
        labels[len(before) : len(after)] = full_ids[len(before) : len(after)]
    target_tokens = sum(label != -100 for label in labels)
    if assistant_messages == 0 or target_tokens == 0:
        raise ValueError(f"assistant教師信号がありません: {record_id}")
    return EncodedRecord(record_id, full_ids, labels)


def percentile(values: Sequence[int], probability: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def token_statistics(records: list[EncodedRecord]) -> dict[str, Any]:
    lengths = [len(record.input_ids) for record in records]
    targets = [sum(label != -100 for label in record.labels) for record in records]
    total_tokens = sum(lengths)
    total_targets = sum(targets)
    return {
        "records": len(records),
        "minimum_tokens": min(lengths, default=0),
        "average_tokens": round(mean(lengths), 2) if lengths else 0,
        "p95_tokens": percentile(lengths, 0.95),
        "maximum_tokens": max(lengths, default=0),
        "assistant_target_tokens": total_targets,
        "all_tokens": total_tokens,
        "assistant_target_ratio": round(total_targets / total_tokens, 6)
        if total_tokens
        else 0,
    }


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("torch", "transformers", "peft", "accelerate", "safetensors"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_fingerprint(
    config: dict[str, Any], dataset_sha256: str, train_ids: list[str], validation_ids: list[str]
) -> str:
    return canonical_hash(
        {
            "config": config,
            "dataset_sha256": dataset_sha256,
            "train_ids": train_ids,
            "validation_ids": validation_ids,
        }
    )


def prepare_run(
    config_path: Path, tokenizer: ChatTokenizer
) -> tuple[
    dict[str, Any],
    Path,
    Path,
    list[EncodedRecord],
    list[EncodedRecord],
    dict[str, Any],
]:
    config, dataset_path, output_dir = resolve_config(config_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"SFT入力がありません: {dataset_path}\n"
            "corpus_creation/v4のfinalizeで生成したv4_sft.jsonlを配置してください。"
        )
    rows = read_jsonl(dataset_path)
    validate_messages(rows, int(config["expected_records"]))
    train_rows, validation_rows = deterministic_split(
        rows,
        float(config["validation_ratio"]),
        int(config["seed"]),
    )
    train = [
        encode_with_assistant_mask(tokenizer, row, int(config["max_length"]))
        for row in train_rows
    ]
    validation = [
        encode_with_assistant_mask(tokenizer, row, int(config["max_length"]))
        for row in validation_rows
    ]
    dataset_sha256 = sha256_file(dataset_path)
    train_ids = [record.record_id for record in train]
    validation_ids = [record.record_id for record in validation]
    fingerprint = build_fingerprint(config, dataset_sha256, train_ids, validation_ids)
    report = {
        "run_fingerprint": fingerprint,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "model_name": config["model_name"],
        "model_revision": config["model_revision"],
        "max_length": config["max_length"],
        "train": token_statistics(train),
        "validation": token_statistics(validation),
        "train_ids": train_ids,
        "validation_ids": validation_ids,
    }
    return config, dataset_path, output_dir, train, validation, report


def resolve_resume(output_dir: Path, resume: str) -> str | bool | None:
    if resume == "none":
        return None
    if resume != "auto":
        checkpoint = Path(resume).resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"checkpointがありません: {checkpoint}")
        return str(checkpoint)
    try:
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as exc:
        raise RuntimeError("transformersが必要です") from exc
    return get_last_checkpoint(str(output_dir)) if output_dir.exists() else None


def assert_resume_compatible(output_dir: Path, fingerprint: str) -> None:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.exists():
        return
    previous = read_json(manifest_path)
    if previous.get("run_fingerprint") != fingerprint:
        raise RuntimeError(
            "既存出力と入力・設定・splitが一致しません。"
            "既存出力を消さず、configのoutput_dirを変更してください。"
        )


def train(config_path: Path, preflight_only: bool, resume: str) -> None:
    config, _, output_dir = resolve_config(config_path)
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("requirements.txtの依存関係を導入してください") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        revision=config["model_revision"],
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("pad_tokenとeos_tokenがありません")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config, dataset_path, output_dir, train_records, validation_records, report = (
        prepare_run(config_path, tokenizer)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "preflight_report.json", report)
    write_json(
        output_dir / "split_manifest.json",
        {
            "run_fingerprint": report["run_fingerprint"],
            "seed": config["seed"],
            "validation_ratio": config["validation_ratio"],
            "train_ids": report["train_ids"],
            "validation_ids": report["validation_ids"],
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if preflight_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPUが必要です")
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("この実行フォルダーはrt_HGの単一GPU用です")
    if bool(config["bf16"]) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("指定GPUはBF16をサポートしていません")

    assert_resume_compatible(output_dir, report["run_fingerprint"])
    checkpoint = resolve_resume(output_dir, resume)
    set_seed(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(config["tf32"])

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        revision=config["model_revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    if bool(config["gradient_checkpointing"]):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["lora_target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    class TokenDataset(Dataset):
        def __init__(self, records: list[EncodedRecord]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            record = self.records[index]
            return {"input_ids": record.input_ids, "labels": record.labels}

    class AssistantOnlyCollator:
        def __init__(self, pad_token_id: int) -> None:
            self.pad_token_id = pad_token_id

        def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
            maximum = max(len(feature["input_ids"]) for feature in features)
            input_ids: list[list[int]] = []
            labels: list[list[int]] = []
            attention_mask: list[list[int]] = []
            for feature in features:
                padding = maximum - len(feature["input_ids"])
                input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
                labels.append(feature["labels"] + [-100] * padding)
                attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

    effective_batch = (
        int(config["per_device_train_batch_size"])
        * int(config["gradient_accumulation_steps"])
    )
    steps_per_epoch = math.ceil(len(train_records) / effective_batch)
    expected_optimizer_steps = math.ceil(
        steps_per_epoch * float(config["num_train_epochs"])
    )
    run_manifest = {
        **report,
        "config_path": str(config_path.resolve()),
        "config": config,
        "git_head": git_head(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "effective_batch_size": effective_batch,
        "steps_per_epoch": steps_per_epoch,
        "expected_optimizer_steps": expected_optimizer_steps,
        "resume_from_checkpoint": checkpoint,
        "completed": False,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        num_train_epochs=float(config["num_train_epochs"]),
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        warmup_ratio=float(config["warmup_ratio"]),
        weight_decay=float(config["weight_decay"]),
        max_grad_norm=float(config["max_grad_norm"]),
        lr_scheduler_type=str(config["lr_scheduler_type"]),
        optim=str(config["optim"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=int(config["logging_steps"]),
        logging_first_step=True,
        save_total_limit=int(config["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bool(config["bf16"]),
        tf32=bool(config["tf32"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        dataloader_num_workers=int(config["dataloader_num_workers"]),
        remove_unused_columns=False,
        report_to=[],
        save_safetensors=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenDataset(train_records),
        eval_dataset=TokenDataset(validation_records),
        data_collator=AssistantOnlyCollator(int(tokenizer.pad_token_id)),
    )
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    evaluation = trainer.evaluate()
    run_manifest.update(
        {
            "completed": True,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "train_metrics": train_result.metrics,
            "evaluation_metrics": evaluation,
            "final_adapter": str(final_dir),
        }
    )
    write_json(output_dir / "run_manifest.json", run_manifest)
    print(f"SFT完了: {final_dir}")


def main() -> None:
    args = parse_args()
    train(args.config, args.preflight_only, args.resume)


if __name__ == "__main__":
    main()
