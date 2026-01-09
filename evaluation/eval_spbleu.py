#!/usr/bin/env python3
# Evaluate BLEU score using sacrebleu with SentencePiece tokenization

import json
import argparse
import sys
import unicodedata
from typing import List, Tuple



# Normalize text using Unicode NFC
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# Load predictions and references from JSONL file
def load_jsonl(input_json: str) -> Tuple[List[str], List[str], int]:
    preds, refs = [], []
    skipped = 0
    with open(input_json, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                skipped += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            pred = str(obj.get("pred", "")).strip()
            ref  = str(obj.get("label", "")).strip()

            if pred and ref:
                preds.append(nfc(pred))
                refs.append(nfc(ref))
            else:
                skipped += 1
    return preds, refs, skipped


# Compute corpus-level BLEU using sacrebleu with SentencePiece tokenization
def compute_spbleu(preds: List[str], refs: List[str], lowercase: bool) -> sacrebleu.metrics.bleu.BLEUScore:
    if len(preds) != len(refs):
        raise ValueError(f"Prediction and reference sample count mismatch: preds={len(preds)} refs={len(refs)}")
    return sacrebleu.corpus_bleu(
        preds,
        [refs],
        tokenize="spm",
        lowercase=lowercase
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute spBLEU (sacrebleu tokenize='spm') on JSONL with fields: pred, label"
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase evaluation")
    args = parser.parse_args()

    preds, refs, skipped = load_jsonl(args.input)
    if not preds:
        print("No samples to evaluate (pred/label empty or parse failed).", file=sys.stderr)
        print(f"Skipped samples: {skipped}")
        sys.exit(2)

    print(f"Loaded: {len(preds)} valid samples, skipped {skipped}.")
    print("Computing spBLEU (sacrebleu tokenize='spm')...\n")

    score = compute_spbleu(preds, refs, lowercase=args.lowercase)
    print(score.format())
    print(f"\nspBLEU: {score.score:.4f}")


if __name__ == "__main__":
    main()
