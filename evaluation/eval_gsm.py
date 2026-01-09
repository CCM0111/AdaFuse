import json
import re
import argparse


# Extract answer from \boxed{...} format
def extract_boxed_answer(text):
    if text is None or not isinstance(text, str):
        return None
    match = re.search(r"\\boxed{([^}]+)}", text)
    return match.group(1).strip() if match else None


# Extract answer from 'the answer is ...' format
def extract_label_answer(text):
    if text is None or not isinstance(text, str):
        return None
    match = re.search(r"[Tt]he answer is\s+(.*)", text)
    return match.group(1).strip() if match else None


# Normalize numeric answer
def normalize(text):
    if text is None:
        return ''
    text = text.lower()
    text = re.sub(r"[^\d\.]", "", text)
    if text == "":
        return ""
    try:
        number = round(float(text), 2)
        return f"{number:.2f}"
    except:
        return text.strip()


# Check if prediction matches the label
def is_match(pred, label):
    norm_pred = normalize(pred)
    norm_label = normalize(label)
    
    if not norm_pred or not norm_label:
        return False
    
    return norm_pred == norm_label or norm_pred in norm_label or norm_label in norm_pred


# Compute GSM8K accuracy
def compute_accuracy(jsonl_path):
    total = 0
    correct = 0
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            
            pred_text = entry.get("pred", "")
            pred_ans = extract_boxed_answer(pred_text)
            if pred_ans is None:
                pred_ans = pred_text
            
            label_text = entry.get("label", "")
            label_ans = extract_label_answer(label_text)
            if label_ans is None:
                label_ans = label_text
            
            total += 1
            if is_match(pred_ans, label_ans):
                correct += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"File: {jsonl_path}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    print("=" * 50)
    
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True)
    args = parser.parse_args()
    
    compute_accuracy(args.pred)


if __name__ == "__main__":
    main()

