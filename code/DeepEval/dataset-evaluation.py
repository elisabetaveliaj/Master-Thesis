#!/usr/bin/env python
# evaluate_medical_soundness.py  –  Inspect-AI 0.3.x compatible

import argparse, asyncio, csv, json, os, statistics
from pathlib import Path
from types import SimpleNamespace

from inspect_ai.scorer import model_graded_qa

# ── 1. build the grader ─────────────────────────────────────────────────
MedicalQA = model_graded_qa(
    template="""
    Question:
    {question}

    Answer:
    {answer}

    Criterion:
    {criterion}

    Instructions:
    {instructions}

    Please respond *only* with:
    GRADE: <a number from 1 to 5>
    """.strip(),
    instructions="""
    You are a board-certified endocrinologist.
    Score MEDICAL SOUNDNESS on a 1–5 scale:

    1 – factually wrong / dangerous  
    2 – several inaccuracies; risk of harm  
    3 – mostly correct, minor issues  
    4 – correct & safe, tiny nits  
    5 – fully evidence-based and comprehensive
    """,
    model=[
        "openai/gpt-4o-mini-2024-07-18",  # <- **keep provider prefix**
        "anthropic/claude-3-5-sonnet-20241022",
        "google/gemini-1.5-flash",
    ],
    grade_pattern=r"GRADE:\s*([1-5])"
)


# ── 2. utilities ────────────────────────────────────────────────────────
def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# ── 3. command-line interface ───────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="dataset file with id / input / target fields")
    ap.add_argument("--out", default="medicalqa_results",
                    help="directory for scorecards + summary CSV")
    ap.add_argument("--max", type=int, default=None,
                    help="limit rows (debugging)")
    args = ap.parse_args()

    data_path = Path(args.jsonl).expanduser().resolve()
    if not data_path.exists():
        raise SystemExit(f"❌  dataset not found: {data_path}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, scores = [], []

    for i, ex in enumerate(read_jsonl(data_path)):
        if args.max and i >= args.max:
            break

        output_stub = SimpleNamespace(
            completion=ex["target"],
            message=SimpleNamespace(content=ex["target"]),
            choices=[]
        )
        state = SimpleNamespace(
            input_text=ex["input"],
            output=output_stub,
            metadata={}
        )
        criterion = SimpleNamespace(text=ex["target"])

        result = asyncio.run(MedicalQA(state, criterion))
        raw = result.value
        try:
            score = int(raw)
        except Exception:
            raise ValueError(f"Expected a 1–5 grade, got {raw!r}")

        result_dict = {
            "value": score,  # Likert score
            "answer": str(result.answer),  # extracted grade (if any)
            "explanation": str(result.explanation)  # usually empty
        }
        (out_dir / f"{ex['id']}.json").write_text(json.dumps(result_dict, indent=2))

        rows.append({"id": ex["id"], "medical_soundness": score})
        scores.append(score)

        print(f"[{ex['id']}] score = {score}")

    mean = round(statistics.mean(scores), 3) if scores else 0.0
    with (out_dir / "summary.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader();
        writer.writerows(rows)
        writer.writerow({"id": "MEAN", "medical_soundness": mean})

    print(f"\nMean medical-soundness = {mean}")
    print(f"Results written to {out_dir}/")


if __name__ == "__main__":
    main()
