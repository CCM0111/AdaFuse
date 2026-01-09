#!/usr/bin/env python3
# Evaluate Natural Questions (NQ) predictions with lenient matching

import argparse
import json
import re
import string
import unicodedata
from pathlib import Path
from typing import Dict, List



# Normalize answer: lowercase, convert number words to digits, remove articles/punctuation/stopwords
def normalize_answer(s: str) -> str:
    s = unicodedata.normalize("NFD", s)

    def replace_number_words_with_digits(text):
        number_word_pattern = (
            r'\b('
            r'(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|'
            r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
            r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
            r'eighty|ninety|hundred|thousand|million|billion|trillion)'
            r'(?:\s+and)?(?:\s+|-)?'
            r'){1,4}\b'
        )
        def _replace(match):
            word = match.group(0)
            word_norm = word.replace("-", " ").replace(" and ", " ")
            try:
                num = w2n.word_to_num(word_norm)
                return str(num)
            except Exception:
                return word
        return re.sub(number_word_pattern, _replace, text, flags=re.IGNORECASE)

    s = replace_number_words_with_digits(s)

    def remove_articles_and_common_stopwords(text):
        pattern = r"\b(a|an|the|and|or|in|on|of|for|to|with|at|by|from|about|as|is|was|were|be|been|are|has|have|had|do|does|did|but|so|if|then|than|that|which|who|whom|whose|because|while|where|when|how|what|why|can|will|would|should|could|shall|may|might|must)\b"
        return re.sub(pattern, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def normalize_possessive_and_plural(text):
        text = re.sub(r"\b(\w+)['']s\b", r"\1", text)
        def _singularize_token(tok):
            if len(tok) >= 4 and tok.endswith("s") and not tok.endswith("ss"):
                return tok[:-1]
            return tok
        return " ".join(_singularize_token(tok) for tok in text.split())

    s = lower(s)
    s = remove_punc(s)
    s = remove_articles_and_common_stopwords(s)
    s = normalize_possessive_and_plural(s)
    s = white_space_fix(s)

    return s


# ---------------------------------------------------------------------------
# 2. Number and name equivalence helpers
# ---------------------------------------------------------------------------

def _num_equiv_any(text: str):
    norm = text.strip().lower()
    if norm.isdigit():
        return int(norm)
    try:
        return w2n.word_to_num(norm)
    except ValueError:
        return None


# Lenient name matching using token subset
def _token_subset_match(a: str, b: str) -> bool:
    toks_a, toks_b = a.split(), b.split()
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
    if n1 is not None and n1 == n2:
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


def is_correct(answers: List[str], prediction: str, is_regex: bool):
    fn = regex_match_score if is_regex else exact_match_score
    return metric_max_over_ground_truths(fn, prediction, answers)


# Core evaluation logic: compute accuracy and track correct predictions
def evaluate_predictions_impl(references: Dict[str, List[str]],
                              predictions: Dict[str, str],
                              is_regex: bool):
    correct_count = 0
    correct_indices = []
    total = len(references)
    for idx, (q, ans_list) in enumerate(references.items(), start=1):
        pred = predictions.get(q, "")
        if is_correct(ans_list, pred, is_regex):
            correct_count += 1
            correct_indices.append(idx)
    missing = len([q for q in references if q not in predictions])
    accuracy = correct_count / total if total > 0 else 0.0
    return {
        'missing_predictions': missing,
        'num_correct': correct_count,
        'num_total': total,
        'accuracy': accuracy,
        'correct_indices': correct_indices
    }


# Extract the last question from prompt text
def _strip_last_prompt_block(raw_q: str) -> str:
    if "Question:" in raw_q:
        last_q = raw_q.split("Question:")[-1]
        if "\nAnswer" in last_q:
            last_q = last_q.split("\nAnswer")[0]
        return last_q.strip()
    return raw_q.strip()


# Load reference answers from JSONL file
def load_references(path: Path, answer_field: str = "answer") -> Dict[str, List[str]]:
    refs: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            ans = example[answer_field]
            if not isinstance(ans, list):
                ans = [ans]
            refs[example["question"].strip().lower()] = ans
    print(f"Found {len(refs)} references in {path}")
    return refs


# Load predictions from JSONL file
def load_predictions(path: Path) -> Dict[str, str]:
    preds: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            raw_q = _strip_last_prompt_block(example["question"])
            q_key = raw_q.lower()
            pred = example["pred"]
            if isinstance(pred, list):
                pred = pred[0]
            preds[q_key] = pred
    print(f"Found {len(preds)} predictions in {path}")
    return preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate NQ predictions with exact-match + extras.")
    parser.add_argument("--pred", required=True, help="Prediction JSONL file")
    parser.add_argument("--ref", required=True, help="Reference JSONL file")
    parser.add_argument("--answer_field", default="answer", help="Answer field name in reference file")
    parser.add_argument("--regex", action="store_true", help="Use regex evaluation (for CuratedTrec)")
    args = parser.parse_args()

    refs = load_references(Path(args.ref), answer_field=args.answer_field)
    preds = load_predictions(Path(args.pred))

    result = evaluate_predictions_impl(refs, preds, is_regex=args.regex)
    print(f"num_q {result['num_total']} correct {result['num_correct']} ratio {result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
