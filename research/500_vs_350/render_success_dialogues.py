"""Render representative high-quality dialogues from the research-350 corpus."""

import json
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "pipelines/corpus_creation/500_empathetic_dialogues.jsonl"
EVALUATIONS = ROOT / "pipelines/corpus_creation/500_dialogue_evaluations.jsonl"
OUTPUT = Path(__file__).resolve().parent / "outputs" / "success_dialogues"
SELECTED_IDS = ["math_train_124", "math_train_232", "math_train_315"]
FONT = Path(r"C:\Windows\Fonts\YuGothM.ttc")


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def clean(text):
    text = text.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    text = text.replace("**", "").replace("\\cdot", "×").replace("\\times", "×").replace("\\%", "%")
    text = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", text)
    text = text.replace("$", "").replace("\\", "")
    text = "".join(char for char in text if char in "\n\t" or ord(char) >= 32 and ord(char) != 127)
    return re.sub(r"[ \t]+", " ", text).strip()


def wrap(draw, text, font, max_width):
    lines = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for char in paragraph.strip():
            candidate = current + char
            if current and draw.textlength(candidate, font=font) > max_width:
                lines.append(current)
                current = char
            else:
                current = candidate
        lines.append(current)
    return lines


def rounded(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def render(record, evaluation, index):
    width, margin = 1600, 96
    title_font = ImageFont.truetype(str(FONT), 48)
    subtitle_font = ImageFont.truetype(str(FONT), 25)
    body_font = ImageFont.truetype(str(FONT), 30)
    label_font = ImageFont.truetype(str(FONT), 24)
    small_font = ImageFont.truetype(str(FONT), 21)
    probe = Image.new("RGB", (width, 100), "white")
    draw = ImageDraw.Draw(probe)
    inner = width - margin * 2

    problem_lines = wrap(draw, clean(record["problem"]), body_font, inner - 56)
    blocks = []
    for turn in record["conversation"]:
        # Both chat bubbles reserve 190 px on one side, plus 60 px inner padding.
        lines = wrap(draw, clean(turn["content"]), body_font, inner - 250)
        height = 44 + len(lines) * 43 + 36
        blocks.append((turn["role"], lines, height))

    header_h = 330 + len(problem_lines) * 42
    height = header_h + sum(h + 28 for _, _, h in blocks) + 110
    image = Image.new("RGB", (width, height), "#F4F7FB")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, width, 190), fill="#17324D")
    draw.text((margin, 47), f"成功対話 {index}｜{record['source_id']}", font=title_font, fill="white")
    scores = evaluation["evaluation"]
    summary = f"総合 {scores['total_score']}/60　数学的正確性 {scores['mathematical_correctness_score']}/10　重大失敗なし"
    draw.text((margin, 122), summary, font=subtitle_font, fill="#CBE5FF")

    y = 225
    draw.text((margin, y), "問題", font=label_font, fill="#31536F")
    y += 46
    problem_h = len(problem_lines) * 42 + 52
    rounded(draw, (margin, y, width - margin, y + problem_h), 22, "white", "#D9E4EE", 2)
    ty = y + 25
    for line in problem_lines:
        draw.text((margin + 28, ty), line, font=body_font, fill="#182632")
        ty += 42
    y += problem_h + 46

    for role, lines, block_h in blocks:
        is_student = role == "student"
        box_x1 = margin + (190 if is_student else 0)
        box_x2 = width - margin - (0 if is_student else 190)
        color = "#E5F1FF" if is_student else "#EAF7EE"
        border = "#9CC8F2" if is_student else "#A8D6B5"
        label = "生徒" if is_student else "教師"
        label_color = "#24689F" if is_student else "#287442"
        draw.text((box_x1 + 24, y), label, font=label_font, fill=label_color)
        y += 38
        rounded(draw, (box_x1, y, box_x2, y + block_h - 38), 24, color, border, 2)
        ty = y + 24
        for line in lines:
            draw.text((box_x1 + 30, ty), line, font=body_font, fill="#182632")
            ty += 43
        y += block_h - 10

    footer = "選定条件：research-350採択済み／総合58点／完了／誤答追認0／不要な反復0"
    draw.text((margin, height - 62), footer, font=small_font, fill="#657786")
    path = OUTPUT / f"{index:02d}_{record['source_id']}.png"
    image.save(path, optimize=True)
    return path


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = {item["source_id"]: item for item in load_jsonl(CORPUS)}
    evaluations = {item["source_id"]: item for item in load_jsonl(EVALUATIONS)}
    paths = [render(records[source_id], evaluations[source_id], i) for i, source_id in enumerate(SELECTED_IDS, 1)]
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
