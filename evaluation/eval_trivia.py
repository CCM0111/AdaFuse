#!/usr/bin/env python3
# Evaluate TriviaQA predictions with lenient matching rules

import argparse
import json
import re
import string
import unicodedata
from pathlib import Path
from typing import Dict, List



# Number words for detecting when to call word2number
NUM_WORDS = {
    "zero","one","two","three","four","five","six","seven","eight","nine",
    "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
    "seventeen","eighteen","nineteen","twenty","thirty","forty","fifty",
    "sixty","seventy","eighty","ninety",
    "hundred","thousand","million","billion","trillion"
}


# Normalize answer: lowercase, remove punctuation/articles/extra spaces
def normalize_answer(s: str) -> str:
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# Try to convert text to int (handles digits and number words)
def _num_equiv_any(text: str):
    norm = text.strip().lower()
    if not norm:
        return None

    if norm.isdigit():
        return int(norm)

    tokens = re.findall(r"[a-zA-Z]+", norm)
    if not any(t in NUM_WORDS for t in tokens):
        return None

    try:
        return w2n.word_to_num(norm)
    except Exception:
        return None


# Lenient name matching using token subset
def _token_subset_match(a: str, b: str) -> bool:
    toks_a = a.split()
    toks_b = b.split()
    if not toks_a or not toks_b:
        return False

    if len(toks_a) <= len(toks_b):
        small, big = toks_a, toks_b
    else:
        small, big = toks_b, toks_a

    if len(small) >= 3 and abs(len(toks_a) - len(toks_b)) > 1:
        return False

    return all(t in big for t in small)


# Check if prediction matches ground truth using exact match or numeric/name equivalence
def exact_match_score(prediction: str, ground_truth: str) -> bool:
    pred_norm, gt_norm = normalize_answer(prediction), normalize_answer(ground_truth)

    if pred_norm == gt_norm:
        return True

    n1, n2 = _num_equiv_any(pred_norm), _num_equiv_any(gt_norm)
    if n1 is not None and n2 is not None and n1 == n2:
        return True

    return _token_subset_match(pred_norm, gt_norm)


def regex_match_score(prediction: str, ground_truth: str) -> bool:
    try:
        regex = re.compile(ground_truth, flags=re.IGNORECASE | re.UNICODE | re.MULTILINE)
        return regex.match(prediction) is not None
    except re.error:
        return False


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def is_correct(answers: List[str], prediction: str, is_regex: bool) -> bool:
    fn = regex_match_score if is_regex else exact_match_score
    return metric_max_over_ground_truths(fn, prediction, answers)


# Core evaluation logic: compute accuracy
def evaluate_predictions_impl(references: Dict[str, List[str]],
                              predictions: Dict[str, str],
                              is_regex: bool):
    correct_count = 0
    total = len(references)
    missing = 0

    for q, ans_list in references.items():
        pred = predictions.get(q, None)
        if pred is None:
            missing += 1
            continue
        if is_correct(ans_list, pred, is_regex):
            correct_count += 1

    acc = correct_count / total if total > 0 else 0.0
    return {
        "missing_predictions": missing,
        "num_correct": correct_count,
        "num_total": total,
        "accuracy": acc,
    }


# Load reference answers from JSONL file
def load_references(path: Path, answer_field: str = "answer") -> Dict[str, List[str]]:
    refs: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            ans = ex[answer_field]
            if not isinstance(ans, list):
                ans = [ans]
            q_key = ex["question"].strip().lower()
            refs[q_key] = ans
    print(f"Found {len(refs)} references in {path}")
    return refs


# Load predictions from JSONL file
def load_predictions(path: Path) -> Dict[str, str]:
    preds: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            q_key = ex["question"].strip().lower()
            pred = ex["pred"]
            if isinstance(pred, list):
                pred = pred[0]
            preds[q_key] = pred
    print(f"Found {len(preds)} predictions in {path}")
    return preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate TriviaQA predictions with exact-match + numeric/name extras.")
    parser.add_argument("--pred", required=True, help="Prediction JSONL file")
    parser.add_argument("--ref", required=True, help="Reference JSONL file")
    parser.add_argument("--answer_field", default="answer", help="Answer field name in reference file")
    parser.add_argument("--regex", action="store_true", help="Use regex evaluation (for CuratedTrec-like datasets)")
    args = parser.parse_args()

    refs = load_references(Path(args.ref), answer_field=args.answer_field)
    preds = load_predictions(Path(args.pred))

    result = evaluate_predictions_impl(refs, preds, is_regex=args.regex)
    print("num_q %d correct %d ratio %.4f" %
          (result["num_total"], result["num_correct"], result["accuracy"]))


if __name__ == "__main__":
    main()
