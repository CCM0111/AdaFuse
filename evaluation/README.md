# Evaluation Scripts

This directory contains evaluation scripts for different NLP tasks supported by AdaFuse.

## Available Evaluators

### 1. GSM8K Evaluation (`eval_gsm.py`)

Evaluates arithmetic reasoning performance on GSM8K dataset.

**Usage:**
```bash
python eval_gsm.py --pred predictions.jsonl
```

**Features:**
- Extracts answers from `\boxed{...}` format
- Handles "the answer is ..." patterns
- Normalizes numeric answers

### 2. Natural Questions Evaluation (`eval_nq.py`)

Evaluates open-domain QA on Natural Questions dataset with lenient matching.

**Usage:**
```bash
python eval_nq.py --pred predictions.jsonl --ref references.jsonl
```

**Features:**
- Lenient answer matching (number words ↔ digits)
- Name matching with token subset
- Case-insensitive comparison

### 3. TriviaQA Evaluation (`eval_trivia.py`)

Evaluates open-domain QA on TriviaQA dataset.

**Usage:**
```bash
python eval_trivia.py --pred predictions.jsonl --ref references.jsonl
```

**Features:**
- Number equivalence matching
- Lenient name matching
- Regex-based matching support

### 4. SQuAD Evaluation (`eval_squad.py`)

Evaluates reading comprehension on SQuAD dataset.

**Usage:**
```bash
python eval_squad.py --pred predictions.jsonl --ref references.jsonl
```

**Features:**
- Exact match scoring
- Lenient answer matching
- Question block cleanup

### 5. BLEU Score Evaluation (`eval_spbleu.py`)

Computes BLEU scores for machine translation tasks using SentencePiece tokenization.

**Usage:**
```bash
python eval_spbleu.py --input predictions.jsonl [--lowercase]
```

**Features:**
- Corpus-level BLEU computation
- SentencePiece tokenization
- Unicode normalization (NFC)

## Output Format

All evaluation scripts expect JSONL format with the following fields:

```json
{
  "question": "What is the capital of France?",
  "answer": "Paris",
  "prediction": "Paris"
}
```

## Common Features

All evaluators support:
- ✅ Lenient matching rules
- ✅ Unicode normalization
- ✅ Case-insensitive comparison
- ✅ Detailed accuracy reporting
- ✅ Batch processing

## Dependencies

```bash
pip install word2number sacrebleu
```

