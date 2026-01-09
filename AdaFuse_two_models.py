import numpy as np
import re
import time
import json
import os
import torch
import torch.nn.functional as F
import argparse
from contextlib import nullcontext
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset
import string
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

from utils.ans_process import *
from utils.collate_fun import *
from utils.extract_response import *


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    softmax_x = exp_x / sum_exp_x
    return softmax_x


def create_qa_collate_fn(prompt_complex):
    def qa_collate_fn(batch):  # TriviaQA / NQ 
        questions, answers = [], []
        for b in batch:
            ques = b["question"]
            prompt_q = prompt_complex + f'\n\nQuestion: {ques}\nAnswer:'
            questions.append(prompt_q)
            answers.append(b["answer"])
        return questions, answers
    return qa_collate_fn


def create_gsm_collate_fn(prompt_complex):
    def gsm_collate_fn(batch):  # GSM8K
        questions, answers = [], []
        for b in batch:
            ques = b["question"]
            prompt_q = prompt_complex + f'\n\nQuestion: {ques}\nLet\'s think step by step\n'
            questions.append(prompt_q)
            answers.append(b["answer"])
        return questions, answers
    return gsm_collate_fn


def count_words_split(text):
    words = text.split()
    return len(words)


def extract_target_question(prompt: str) -> str:
    m = re.findall(r"\nQuestion:\s*(.*?)\nAnswer:", prompt, flags=re.S | re.I)
    if m:
        return m[-1].strip()
    return prompt.strip()


def compute_top1_margin(logits):
    probs = F.softmax(logits, dim=-1)
    top_probs, _ = torch.topk(probs, k=2, dim=-1)
    margin = (top_probs[0, 0] - top_probs[0, 1]).item()
    return margin


def check_boundary_gate(first_word, second_word):
    if first_word.strip()[-1:].isdigit() and second_word.strip()[:1].isdigit():
        return False
    
    if '"' in first_word or "'" in first_word:
        return False
    
    if '(' in first_word or ')' in first_word:
        return False
    
    return True


def extract_word_tokens_from_generated(tokenizer, generated_tokens, target_word):
    if not generated_tokens:
        return generated_tokens
    
    target_word = target_word.strip()
    if not target_word:
        return generated_tokens
    
    last_valid_tokens = None
    last_valid_text = ""
    
    for i in range(1, len(generated_tokens) + 1):
        partial_tokens = generated_tokens[:i]
        decoded_text = tokenizer.decode(partial_tokens, skip_special_tokens=False).strip()
        
        if not decoded_text:
            continue
        
        if decoded_text == target_word:
            return partial_tokens
        
        if decoded_text.lower() == target_word.lower():
            return partial_tokens
        
        if target_word.startswith(decoded_text):
            last_valid_tokens = partial_tokens
            last_valid_text = decoded_text
            continue
        
        if target_word.lower().startswith(decoded_text.lower()):
            last_valid_tokens = partial_tokens
            last_valid_text = decoded_text
            continue
        
        if decoded_text.startswith(target_word) and len(decoded_text) > len(target_word):
            if last_valid_tokens is not None:
                return last_valid_tokens
            else:
                if i > 1:
                    return generated_tokens[:i-1]
                else:
                    return partial_tokens
        
        if decoded_text.lower().startswith(target_word.lower()) and len(decoded_text) > len(target_word):
            if last_valid_tokens is not None:
                return last_valid_tokens
            else:
                if i > 1:
                    return generated_tokens[:i-1]
                else:
                    return partial_tokens
    
    if last_valid_tokens is not None:
        return last_valid_tokens
    
    return generated_tokens


def generate_candidate_spans(model, tokenizer, prompt, span_length, gen_config, beam_size, theta_delta=0.7, max_words=3, forward_count=None):
    THETA_DELTA = theta_delta
    
    if forward_count is None:
        forward_count = {"generate": 0, "ppl": 0}
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].size(1)

    if beam_size == 1:
        first_cfg_single = gen_config.__class__(
            num_beams=1,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=gen_config.pad_token_id,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        forward_count["generate"] += 1
        with torch.amp.autocast(device_type="cuda"):
            first_out_single = model.generate(
                **inputs,
                generation_config=first_cfg_single,
                return_dict_in_generate=True,
            )
        forward_count["generate"] += 1
        forward_count["generate"] += 1

        first_token_logits_single = first_out_single.scores[0][0:1]
        Delta_k_single = compute_top1_margin(first_token_logits_single)
        is_stable_single = (Delta_k_single >= THETA_DELTA)
        
        first_seq_single = first_out_single.sequences[0]
        
        continue_cfg_single = gen_config.__class__(
            num_beams=1,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=gen_config.pad_token_id,
            max_new_tokens=6,
            return_dict_in_generate=True,
        )
        
        continue_inputs_single = {
            "input_ids": first_seq_single.unsqueeze(0),
            "attention_mask": torch.ones_like(first_seq_single.unsqueeze(0)),
        }
        
        with torch.amp.autocast(device_type="cuda"):
            continue_out_single = model.generate(
                **continue_inputs_single,
                generation_config=continue_cfg_single,
                return_dict_in_generate=True,
            )
        forward_count["generate"] += 6  # 6 tokens * beam=1
        forward_count["generate"] += 6

        first_word_seq_single = continue_out_single.sequences[0]
        first_word_ids = first_word_seq_single[prompt_len:].tolist()

        spans = []
        
        if not first_word_ids:
            print(f"[DEBUG beam=1] first_word_ids is empty after generating 7 tokens. Returning empty spans.", flush=True)
            return spans
        
        first_word_txt_raw = tokenizer.decode(first_word_ids, skip_special_tokens=False)
        words_count = len(first_word_txt_raw.split())
        
        if words_count < 2 and "</s>" not in first_word_txt_raw and "question:" not in first_word_txt_raw.lower() and "note:" not in first_word_txt_raw.lower():
            print(f"[DEBUG beam=1] Only {words_count} word(s) from 7 tokens: '{first_word_txt_raw}'. Generating 5 more tokens...", flush=True)
            extend_cfg_single = gen_config.__class__(
                num_beams=1,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=gen_config.pad_token_id,
                max_new_tokens=5,
                return_dict_in_generate=True,
            )
            forward_count["generate"] += 5
            extend_inputs_single = {
                "input_ids": first_word_seq_single.unsqueeze(0),
                "attention_mask": torch.ones_like(first_word_seq_single.unsqueeze(0)),
            }
            with torch.amp.autocast(device_type="cuda"):
                extend_out_single = model.generate(
                    **extend_inputs_single,
                    generation_config=extend_cfg_single,
                    return_dict_in_generate=True,
                )
            forward_count["generate"] += 5  # 5 tokens * beam=1
            forward_count["generate"] += 5
            extended_seq = extend_out_single.sequences[0]
            first_word_ids = extended_seq[prompt_len:].tolist()
            first_word_txt_raw = tokenizer.decode(first_word_ids, skip_special_tokens=False)
            print(f"[DEBUG beam=1] After extension ({len(first_word_ids)} tokens): '{first_word_txt_raw}'", flush=True)
        
        words_list = first_word_txt_raw.split()
        
        valid_words = [w for w in words_list if w.strip() and not all(ch in string.punctuation for ch in w)]
        if valid_words:
            first_word_txt_single = " ".join(valid_words[:span_length])
        elif words_list:
            first_word_txt_single = " ".join(words_list[:span_length])
        else:
            first_word_txt_single = ""
        
        if not first_word_txt_single.strip() or all(ch in string.punctuation for ch in first_word_txt_single):
            print(f"[DEBUG beam=1] First word still only punctuation: '{first_word_txt_raw}'", flush=True)
        
        first_word_only_ids = extract_word_tokens_from_generated(tokenizer, first_word_ids, first_word_txt_single)
        first_word_only_seq = torch.cat([
            inputs["input_ids"][0, :prompt_len],
            torch.tensor(first_word_only_ids, dtype=torch.long, device=model.device)
        ])
        
        if not is_stable_single:
            if first_word_txt_single.strip() and not all(ch in string.punctuation for ch in first_word_txt_single):
                spans.append((first_word_txt_single, first_word_only_ids))
            else:
                print(f"[DEBUG beam=1] First word filtered (empty or all punctuation): '{first_word_txt_single}'", flush=True)
        else:
            second_token_cfg_single = gen_config.__class__(
                num_beams=1,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=gen_config.pad_token_id,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            second_token_inputs_single = {
                "input_ids": first_word_only_seq.unsqueeze(0),
                "attention_mask": torch.ones_like(first_word_only_seq.unsqueeze(0)),
            }
            
            with torch.amp.autocast(device_type="cuda"):
                second_token_out_single = model.generate(
                    **second_token_inputs_single,
                    generation_config=second_token_cfg_single,
                    return_dict_in_generate=True,
                )
            forward_count["generate"] += 1  # 1 token * beam=1
            
            second_token_logits_single = second_token_out_single.scores[0][0:1]
            Delta_k_second_single = compute_top1_margin(second_token_logits_single)
            is_stable_second_single = (Delta_k_second_single >= THETA_DELTA)
            
            second_token_seq_single = second_token_out_single.sequences[0]
            
            if not is_stable_second_single:
                second_word_cfg_single = gen_config.__class__(
                    num_beams=1,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=gen_config.pad_token_id,
                    max_new_tokens=6,
                    return_dict_in_generate=True,
                )
            else:
                second_word_cfg_single = gen_config.__class__(
                    num_beams=1,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=gen_config.pad_token_id,
                    max_new_tokens=5,
                    return_dict_in_generate=True,
                )
            
            second_word_inputs_single = {
                "input_ids": second_token_seq_single.unsqueeze(0),
                "attention_mask": torch.ones_like(second_token_seq_single.unsqueeze(0)),
            }
            
            with torch.amp.autocast(device_type="cuda"):
                second_word_out_single = model.generate(
                    **second_word_inputs_single,
                    generation_config=second_word_cfg_single,
                    return_dict_in_generate=True,
                )
            forward_count["generate"] += 5  # 5 tokens * beam=1
            
            full_seq_single = second_word_out_single.sequences[0]
            full_ids_single = full_seq_single[len(first_word_only_seq):].tolist()
            
            if not full_ids_single:
                print(f"[DEBUG beam=1] full_ids_single is empty, falling back to first word only", flush=True)
                if first_word_txt_single.strip() and not all(ch in string.punctuation for ch in first_word_txt_single):
                    spans.append((first_word_txt_single, first_word_only_ids))
                else:
                    print(f"[DEBUG beam=1] First word also filtered: '{first_word_txt_single}'", flush=True)
            else:
                full_txt_single = tokenizer.decode(full_ids_single, skip_special_tokens=False)
                words_count_w2w3 = len(full_txt_single.split())
                
                min_words_needed = 3 if is_stable_second_single else 2
                
                if words_count_w2w3 < min_words_needed and "</s>" not in full_txt_single and "question:" not in full_txt_single.lower() and "note:" not in full_txt_single.lower():
                    print(f"[DEBUG beam=1] w2(+w3): Only {words_count_w2w3} word(s) from {len(full_ids_single)} tokens: '{full_txt_single}'. Generating 5 more tokens...", flush=True)
                    extend_w2w3_cfg = gen_config.__class__(
                        num_beams=1,
                        num_return_sequences=1,
                        do_sample=False,
                        pad_token_id=gen_config.pad_token_id,
                        max_new_tokens=5,
                        return_dict_in_generate=True,
                    )
                    forward_count["generate"] += 5
                    extend_w2w3_inputs = {
                        "input_ids": full_seq_single.unsqueeze(0),
                        "attention_mask": torch.ones_like(full_seq_single.unsqueeze(0)),
                    }
                    with torch.amp.autocast(device_type="cuda"):
                        extend_w2w3_out = model.generate(
                            **extend_w2w3_inputs,
                            generation_config=extend_w2w3_cfg,
                            return_dict_in_generate=True,
                        )
                    forward_count["generate"] += 5  # 5 tokens * beam=1
                    extended_w2w3_seq = extend_w2w3_out.sequences[0]
                    full_ids_single = extended_w2w3_seq[len(first_word_only_seq):].tolist()
                    full_txt_single = tokenizer.decode(full_ids_single, skip_special_tokens=False)
                    print(f"[DEBUG beam=1] w2(+w3): After extension ({len(full_ids_single)} tokens): '{full_txt_single}'", flush=True)
                
                words_single = full_txt_single.split()
                
                if not is_stable_second_single:
                    if len(words_single) >= 1:
                        second_word = words_single[0]
                        span_txt_single = first_word_txt_single + " " + second_word
                        combined_tokens = first_word_only_ids + full_ids_single
                        span_ids = extract_word_tokens_from_generated(tokenizer, combined_tokens, span_txt_single)
                        
                        if span_txt_single.strip() and not all(ch in string.punctuation for ch in span_txt_single):
                            spans.append((span_txt_single, span_ids))
                    else:
                        if first_word_txt_single.strip() and not all(ch in string.punctuation for ch in first_word_txt_single):
                            spans.append((first_word_txt_single, first_word_only_ids))
                else:
                    if len(words_single) >= 2:
                        second_word = words_single[0]
                        third_word = words_single[1]
                        span_txt_single = first_word_txt_single + " " + second_word + " " + third_word
                        combined_tokens = first_word_only_ids + full_ids_single
                        span_ids = extract_word_tokens_from_generated(tokenizer, combined_tokens, span_txt_single)
                        
                        if span_txt_single.strip() and not all(ch in string.punctuation for ch in span_txt_single):
                            spans.append((span_txt_single, span_ids))
                    elif len(words_single) >= 1:
                        second_word = words_single[0]
                        span_txt_single = first_word_txt_single + " " + second_word
                        combined_tokens = first_word_only_ids + full_ids_single
                        span_ids = extract_word_tokens_from_generated(tokenizer, combined_tokens, span_txt_single)
                        
                        if span_txt_single.strip() and not all(ch in string.punctuation for ch in span_txt_single):
                            spans.append((span_txt_single, span_ids))
                    else:
                        if first_word_txt_single.strip() and not all(ch in string.punctuation for ch in first_word_txt_single):
                            spans.append((first_word_txt_single, first_word_only_ids))

        return spans

    first_cfg_check = gen_config.__class__(
        num_beams=1,
        num_return_sequences=1,
        do_sample=False,
        pad_token_id=gen_config.pad_token_id,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
    )
    
    with torch.amp.autocast(device_type="cuda"):
        first_out_check = model.generate(
            **inputs,
            generation_config=first_cfg_check,
            return_dict_in_generate=True,
        )
    forward_count["generate"] += 1  # 1 token * beam=1

    first_token_logits = first_out_check.scores[0][0:1]
    Delta_k = compute_top1_margin(first_token_logits)
    is_stable = (Delta_k >= THETA_DELTA)

    spans = []
    
    if is_stable:
        first_seq_single = first_out_check.sequences[0]
        
        continue_cfg_single = gen_config.__class__(
            num_beams=1,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=gen_config.pad_token_id,
            max_new_tokens=6,
            return_dict_in_generate=True,
        )
        
        continue_inputs_single = {
            "input_ids": first_seq_single.unsqueeze(0),
            "attention_mask": torch.ones_like(first_seq_single.unsqueeze(0)),
        }
        
        with torch.amp.autocast(device_type="cuda"):
            continue_out_single = model.generate(
                **continue_inputs_single,
                generation_config=continue_cfg_single,
                return_dict_in_generate=True,
            )
        forward_count["generate"] += 6  # 6 tokens * beam=1
        
        first_word_seq_single = continue_out_single.sequences[0]
        first_word_ids = first_word_seq_single[prompt_len:].tolist()
        
        if not first_word_ids:
            print(f"[DEBUG beam=3] first_word_ids is empty after generating 7 tokens. Returning empty spans.", flush=True)
            return spans
        
        first_word_txt_raw = tokenizer.decode(first_word_ids, skip_special_tokens=False)
        words_count = len(first_word_txt_raw.split())
        
        if words_count < 2 and "</s>" not in first_word_txt_raw and "question:" not in first_word_txt_raw.lower() and "note:" not in first_word_txt_raw.lower():
            print(f"[DEBUG beam=3] Only {words_count} word(s) from 7 tokens: '{first_word_txt_raw}'. Generating 5 more tokens...", flush=True)
            extend_cfg_single = gen_config.__class__(
                num_beams=1,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=gen_config.pad_token_id,
                max_new_tokens=5,
                return_dict_in_generate=True,
            )
            forward_count["generate"] += 5
            extend_inputs_single = {
                "input_ids": first_word_seq_single.unsqueeze(0),
                "attention_mask": torch.ones_like(first_word_seq_single.unsqueeze(0)),
            }
            with torch.amp.autocast(device_type="cuda"):
                extend_out_single = model.generate(
                    **extend_inputs_single,
                    generation_config=extend_cfg_single,
                    return_dict_in_generate=True,
                )
            forward_count["generate"] += 5  # 5 tokens * beam=1
            extended_seq = extend_out_single.sequences[0]
            first_word_ids = extended_seq[prompt_len:].tolist()
            first_word_txt_raw = tokenizer.decode(first_word_ids, skip_special_tokens=False)
            print(f"[DEBUG beam=3] After extension ({len(first_word_ids)} tokens): '{first_word_txt_raw}'", flush=True)
        
        words_list = first_word_txt_raw.split()
        
        valid_words = [w for w in words_list if w.strip() and not all(ch in string.punctuation for ch in w)]
        if valid_words:
            first_word_txt_single = " ".join(valid_words[:span_length])
        elif words_list:
            first_word_txt_single = " ".join(words_list[:span_length])
        else:
            first_word_txt_single = ""
        
        if not first_word_txt_single.strip() or all(ch in string.punctuation for ch in first_word_txt_single):
            print(f"[DEBUG beam=3] First word still only punctuation: '{first_word_txt_raw}'", flush=True)
        
        first_word_7_tokens = first_word_ids
        
        first_word_only_ids = extract_word_tokens_from_generated(tokenizer, first_word_ids, first_word_txt_single)
        first_word_only_seq = torch.cat([
            inputs["input_ids"][0, :prompt_len],
            torch.tensor(first_word_only_ids, dtype=torch.long, device=model.device)
        ])
        
        second_token_cfg_single = gen_config.__class__(
            num_beams=1,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=gen_config.pad_token_id,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        second_token_inputs_single = {
            "input_ids": first_word_only_seq.unsqueeze(0),
            "attention_mask": torch.ones_like(first_word_only_seq.unsqueeze(0)),
        }
        
        with torch.amp.autocast(device_type="cuda"):
            second_token_out_single = model.generate(
                **second_token_inputs_single,
                generation_config=second_token_cfg_single,
                return_dict_in_generate=True,
            )
        forward_count["generate"] += 1  # 1 tokens * beam=1
        
        second_token_logits_single = second_token_out_single.scores[0][0:1]
        Delta_k_second_single = compute_top1_margin(second_token_logits_single)
        is_stable_second_single = (Delta_k_second_single >= THETA_DELTA)
        
        second_token_seq_single = second_token_out_single.sequences[0]
        
        second_word_cfg_single = gen_config.__class__(
            num_beams=1,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=gen_config.pad_token_id,
            max_new_tokens=6,
            return_dict_in_generate=True,
        )
        
        second_word_inputs_single = {
            "input_ids": second_token_seq_single.unsqueeze(0),
            "attention_mask": torch.ones_like(second_token_seq_single.unsqueeze(0)),
        }
        
        with torch.amp.autocast(device_type="cuda"):
            second_word_out_single = model.generate(
                **second_word_inputs_single,
                generation_config=second_word_cfg_single,
                return_dict_in_generate=True,
            )
        forward_count["generate"] += 6  # 6 tokens * beam=1
        
        second_word_seq_single = second_word_out_single.sequences[0]
        second_word_ids_full = second_word_seq_single[len(first_word_only_seq):].tolist()
        
        if not second_word_ids_full:
            if first_word_txt_single.strip() and not all(ch in string.punctuation for ch in first_word_txt_single):
                spans.append((first_word_txt_single, first_word_7_tokens))
        else:
            second_word_txt_raw = tokenizer.decode(second_word_ids_full, skip_special_tokens=False)
            words_count_w2 = len(second_word_txt_raw.split())
            
            if words_count_w2 < 2 and "</s>" not in second_word_txt_raw and "question:" not in second_word_txt_raw.lower() and "note:" not in second_word_txt_raw.lower():
                print(f"[DEBUG beam=3] w2: Only {words_count_w2} word(s) from 7 tokens: '{second_word_txt_raw}'. Generating 5 more tokens...", flush=True)
                extend_w2_cfg = gen_config.__class__(
                    num_beams=1,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=gen_config.pad_token_id,
                    max_new_tokens=5,
                    return_dict_in_generate=True,
                )
                forward_count["generate"] += 5
                extend_w2_inputs = {
                    "input_ids": second_word_seq_single.unsqueeze(0),
                    "attention_mask": torch.ones_like(second_word_seq_single.unsqueeze(0)),
                }
                with torch.amp.autocast(device_type="cuda"):
                    extend_w2_out = model.generate(
                        **extend_w2_inputs,
                        generation_config=extend_w2_cfg,
                        return_dict_in_generate=True,
                    )
                forward_count["generate"] += 5  # 5 tokens * beam=1
                extended_w2_seq = extend_w2_out.sequences[0]
                second_word_ids_full = extended_w2_seq[len(first_word_only_seq):].tolist()
                second_word_txt_raw = tokenizer.decode(second_word_ids_full, skip_special_tokens=False)
                print(f"[DEBUG beam=3] w2: After extension ({len(second_word_ids_full)} tokens): '{second_word_txt_raw}'", flush=True)
            
            words_list = second_word_txt_raw.split()
            
            if len(words_list) >= 1:
                second_word = words_list[0]
                
                if not is_stable_second_single:
                    span_txt_single = first_word_txt_single + " " + second_word
                    span_ids = first_word_only_ids + second_word_ids_full
                    
                    if span_txt_single.strip() and not all(ch in string.punctuation for ch in span_txt_single):
                        spans.append((span_txt_single, span_ids))
                else:
                        first_two_words_txt = first_word_txt_single + " " + second_word
                        combined_tokens = first_word_only_ids + second_word_ids_full
                        first_two_words_ids = extract_word_tokens_from_generated(tokenizer, combined_tokens, first_two_words_txt)
                        first_two_words_seq = torch.cat([
                            inputs["input_ids"][0, :prompt_len],
                            torch.tensor(first_two_words_ids, dtype=torch.long, device=model.device)
                        ])
                        
                        third_token_cfg_beam3 = gen_config.__class__(
                            num_beams=beam_size,
                            num_return_sequences=beam_size,
                            do_sample=False,
                            pad_token_id=gen_config.pad_token_id,
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                        )
                        
                        third_token_inputs_beam3 = {
                            "input_ids": first_two_words_seq.unsqueeze(0),
                            "attention_mask": torch.ones_like(first_two_words_seq.unsqueeze(0)),
                        }
                        
                        with torch.amp.autocast(device_type="cuda"):
                            third_token_out_beam3 = model.generate(
                                **third_token_inputs_beam3,
                                generation_config=third_token_cfg_beam3,
                                return_dict_in_generate=True,
                            )
                        forward_count["generate"] += 3  # 1 tokens * beam=3
                        
                        for beam_seq in third_token_out_beam3.sequences:
                            third_word_cfg = gen_config.__class__(
                                num_beams=1,
                                num_return_sequences=1,
                                do_sample=False,
                                pad_token_id=gen_config.pad_token_id,
                                max_new_tokens=6,
                                return_dict_in_generate=True,
                            )
                            
                            third_word_inputs = {
                                "input_ids": beam_seq.unsqueeze(0),
                                "attention_mask": torch.ones_like(beam_seq.unsqueeze(0)),
                            }
                            
                            with torch.amp.autocast(device_type="cuda"):
                                third_word_out = model.generate(
                                    **third_word_inputs,
                                    generation_config=third_word_cfg,
                                    return_dict_in_generate=True,
                                )
                            forward_count["generate"] += 6  # 6 tokens * beam=1
                            
                            third_word_seq = third_word_out.sequences[0]
                            third_word_ids_full = third_word_seq[len(first_two_words_seq):].tolist()
                            
                            if not third_word_ids_full:
                                continue
                            
                            third_word_txt_raw = tokenizer.decode(third_word_ids_full, skip_special_tokens=False)
                            words_count_w3 = len(third_word_txt_raw.split())
                            
                            if words_count_w3 < 2 and "</s>" not in third_word_txt_raw and "question:" not in third_word_txt_raw.lower() and "note:" not in third_word_txt_raw.lower():
                                print(f"[DEBUG beam=3] w3: Only {words_count_w3} word(s) from 7 tokens: '{third_word_txt_raw}'. Generating 5 more tokens...", flush=True)
                                extend_w3_cfg = gen_config.__class__(
                                    num_beams=1,
                                    num_return_sequences=1,
                                    do_sample=False,
                                    pad_token_id=gen_config.pad_token_id,
                                    max_new_tokens=5,
                                    return_dict_in_generate=True,
                                )
                                forward_count["generate"] += 5
                                extend_w3_inputs = {
                                    "input_ids": third_word_seq.unsqueeze(0),
                                    "attention_mask": torch.ones_like(third_word_seq.unsqueeze(0)),
                                }
                                with torch.amp.autocast(device_type="cuda"):
                                    extend_w3_out = model.generate(
                                        **extend_w3_inputs,
                                        generation_config=extend_w3_cfg,
                                        return_dict_in_generate=True,
                                    )
                                forward_count["generate"] += 5  # 5 tokens * beam=1
                                extended_w3_seq = extend_w3_out.sequences[0]
                                third_word_ids_full = extended_w3_seq[len(first_two_words_seq):].tolist()
                                third_word_txt_raw = tokenizer.decode(third_word_ids_full, skip_special_tokens=False)
                                print(f"[DEBUG beam=3] w3: After extension ({len(third_word_ids_full)} tokens): '{third_word_txt_raw}'", flush=True)
                            
                            words_list_three = third_word_txt_raw.split()
                            
                            if len(words_list_three) >= 1:
                                third_word = words_list_three[0]
                                
                                span_txt_single = first_word_txt_single + " " + second_word + " " + third_word
                                span_ids = first_two_words_ids + third_word_ids_full
                                
                                if span_txt_single.strip() and not all(ch in string.punctuation for ch in span_txt_single):
                                    spans.append((span_txt_single, span_ids))
            else:
                if first_word_txt_single.strip() and not all(ch in string.punctuation for ch in first_word_txt_single):
                    spans.append((first_word_txt_single, first_word_7_tokens))
    
    else:
        first_cfg_beam3 = gen_config.__class__(
            num_beams=beam_size,
            num_return_sequences=beam_size,
            do_sample=False,
            pad_token_id=gen_config.pad_token_id,
            max_new_tokens=1,
            return_dict_in_generate=True,
        )
        
        with torch.amp.autocast(device_type="cuda"):
            first_out_beam3 = model.generate(
                **inputs,
                generation_config=first_cfg_beam3,
                return_dict_in_generate=True,
            )
        forward_count["generate"] += 3  # 1 tokens * beam=3
        
        for beam_seq in first_out_beam3.sequences:
            continue_cfg = gen_config.__class__(
                num_beams=1,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=gen_config.pad_token_id,
                max_new_tokens=6,
                return_dict_in_generate=True,
            )
            
            continue_inputs = {
                "input_ids": beam_seq.unsqueeze(0),
                "attention_mask": torch.ones_like(beam_seq.unsqueeze(0)),
            }
            
            with torch.amp.autocast(device_type="cuda"):
                continue_out = model.generate(
                    **continue_inputs,
                    generation_config=continue_cfg,
                    return_dict_in_generate=True,
                )
            forward_count["generate"] += 6  # 6 tokens * beam=1
            
            final_seq = continue_out.sequences[0]
            new_ids = final_seq[prompt_len:].tolist()
            
            if not new_ids:
                continue
            
            gen_txt = tokenizer.decode(new_ids, skip_special_tokens=False)
            words_count = len(gen_txt.split())
            
            if words_count < 2 and "</s>" not in gen_txt and "question:" not in gen_txt.lower() and "note:" not in gen_txt.lower():
                extend_cfg = gen_config.__class__(
                    num_beams=1,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=gen_config.pad_token_id,
                    max_new_tokens=5,
                    return_dict_in_generate=True,
                )
                forward_count["generate"] += 5
                extend_inputs = {
                    "input_ids": final_seq.unsqueeze(0),
                    "attention_mask": torch.ones_like(final_seq.unsqueeze(0)),
                }
                with torch.amp.autocast(device_type="cuda"):
                    extend_out = model.generate(
                        **extend_inputs,
                        generation_config=extend_cfg,
                        return_dict_in_generate=True,
                    )
                forward_count["generate"] += 5  # 5 tokens * beam=1
                extended_seq = extend_out.sequences[0]
                new_ids = extended_seq[prompt_len:].tolist()
                gen_txt = tokenizer.decode(new_ids, skip_special_tokens=False)
            
            gen_words = gen_txt.split()
            valid_words = [w for w in gen_words if w.strip() and not all(ch in string.punctuation for ch in w)]
            
            if valid_words:
                span_txt = " ".join(valid_words[:span_length])
            elif gen_words:
                span_txt = " ".join(gen_words[:span_length])
            else:
                continue
            
            span_ids = new_ids
            
            if span_txt.strip() and not all(ch in string.punctuation for ch in span_txt):
                spans.append((span_txt, span_ids))

    return spans


@torch.no_grad()
def compute_span_perplexity(model, tokenizer, prefix, span, forward_count=None):
    if forward_count is not None:
        forward_count["ppl"] += 1
    
    inputs = tokenizer(prefix, return_tensors='pt').to(model.device)
    inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.tensor([span], dtype = inputs['input_ids'].dtype, device = model.device)], dim = -1)
    inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones(1, len(span), dtype = inputs['attention_mask'].dtype, device = model.device)], dim = -1)
    
    
    outputs = model(**inputs)
    logits = F.log_softmax(outputs.logits,dim=-1)  
    
    prefix_len = inputs["input_ids"].shape[1] - len(span)
    
    shift_logits = logits[:, prefix_len-1:-1, :]
    shift_labels = inputs["input_ids"][:, prefix_len:]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())
   
    avg_loss_per_sample = loss.mean(dim=1)
    ppl = avg_loss_per_sample.item()#torch.exp(avg_loss_per_sample).item()
    return ppl


def filter_perplexities(ppl_values, lambda_val=10.0):

    max_val = max(ppl_values)
    min_val = min(ppl_values)
    
    if min_val == 0:
        ratio = float('inf')
    else:
        ratio = max_val / min_val

    if ratio > lambda_val:
        filtered = [v for v in ppl_values if v != max_val and v != min_val]
        if not filtered:  
            filtered = [v for v in ppl_values if v != max_val]
    else:
        filtered = ppl_values

    return filtered


def clean_text_before_question(text):
    match = re.search(r"(.*?)(?:\bquestion\s*:|\bnote\s*:|<)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def extract_boxed_answer(text):
    match = re.search(r'(.*?\\?boxed\s*\{[^}]*\})', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return text.strip()


def ensemble_decoding_worker(gpu_id, start_idx, end_idx, args, prompt_complex):

    forward_count_model1 = {"generate": 0, "ppl": 0}
    forward_count_model2 = {"generate": 0, "ppl": 0}

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    
    print(f"GPU {gpu_id}: Processing samples {start_idx} to {end_idx-1}")
    

    model1 = AutoModelForCausalLM.from_pretrained(
        args.model_path1, 
        torch_dtype=torch.float16,
        use_auth_token=True,
        device_map={"": gpu_id}
    ).eval()
    
    model2 = AutoModelForCausalLM.from_pretrained(
        args.model_path2, 
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_auth_token=True,
        device_map={"": gpu_id}
    ).eval()

    tokenizer1 = AutoTokenizer.from_pretrained(args.model_path1)
    tokenizer2 = AutoTokenizer.from_pretrained(args.model_path2, trust_remote_code=True)

    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer2.pad_token = tokenizer2.eos_token
    tokenizer1.padding_side = "left"
    tokenizer2.padding_side = "left"

    generation_config1 = GenerationConfig(
        num_beams=args.beam_size,
        num_return_sequences=args.beam_size,
        do_sample=False,
        pad_token_id=tokenizer1.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        output_hidden_states=True,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
        use_cache=True,
    )
    generation_config2 = GenerationConfig(
        num_beams=args.beam_size,
        num_return_sequences=args.beam_size,
        do_sample=False,
        pad_token_id=tokenizer2.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        output_hidden_states=True,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
        use_cache=True,
    )


    test_dataset = load_dataset("json", data_files=args.test_set, split='train')
    subset_dataset = Subset(test_dataset, range(start_idx, end_idx))
    
    if 'gsm' in args.test_set.lower():
        collate_fn = create_gsm_collate_fn(prompt_complex)
    else:
        collate_fn = create_qa_collate_fn(prompt_complex)
    
 
    ds_loader = DataLoader(
        subset_dataset,
        batch_size=args.per_device_batch_size,
        collate_fn=collate_fn,
        num_workers=0  
    )

    all_results = []
    

    log_file = f"span_trace_log_gpu{gpu_id}.txt"
    
    total_batches = len(ds_loader)
    batch_count = 0
    
    for questions, answers in ds_loader:
        for idx, prompt in enumerate(questions):
            orig_question = extract_target_question(prompt) 
            current_prefix = prompt
            span_trace = []
            
            for i in range(args.max_span_rounds):
                candidates1 = generate_candidate_spans(model1, tokenizer1, current_prefix,
                                                       args.span_length, generation_config1, args.beam_size,
                                                       args.theta_delta, args.max_words)
                candidates2 = generate_candidate_spans(model2, tokenizer2, current_prefix,
                                                       args.span_length, generation_config2, args.beam_size,
                                                       args.theta_delta, args.max_words)
                
                debug_info = {
                    "model1_count": len(candidates1),
                    "model2_count": len(candidates2),
                    "model1_spans": [s for s, _ in candidates1],
                    "model2_spans": [s for s, _ in candidates2]
                }
                
                cands = dict()
                for s, tok_ids in candidates1:
                    cands.setdefault(s, [None, None])
                    cands[s][0] = tok_ids
                for s, tok_ids in candidates2:
                    cands.setdefault(s, [None, None])
                    cands[s][1] = tok_ids



                with open(log_file, "a", encoding="utf-8") as logf:
                   
                    logf.write(f"\nRound {i + 1} — Candidate Spans & PPL:\n")

                  
                    for model_tag, cand in [("Model1", candidates1), ("Model2", candidates2)]:
                        for beam_idx, (span_txt, tok_ids) in enumerate(cand, 1):
                            logf.write(
                                f'{model_tag}: Beam{beam_idx}: TokenIds: {tok_ids}  Span: "{span_txt}"\n'
                            )

                  
                    ppl_scores = []
                    nan_count = 0
                    for span_txt, (span1, span2) in cands.items():      
                        if span1 is not None:
                            tok_ids1 = span1
                        else:
                            tok_ids1 = tokenizer1(span_txt, add_special_tokens=False)["input_ids"]
                        if span2 is not None:
                            tok_ids2 = span2
                        else:
                            tok_ids2 = tokenizer2(span_txt, add_special_tokens=False)["input_ids"]
                        ppl1 = compute_span_perplexity(model1, tokenizer1, current_prefix, tok_ids1, forward_count_model1)
                        ppl2 = compute_span_perplexity(model2, tokenizer2, current_prefix, tok_ids2, forward_count_model2)
                        if np.isnan(ppl1) and np.isnan(ppl2):
                            nan_count += 1
                            logf.write(f'Span: "{span_txt}" - SKIPPED (Both PPLs are NaN)\n')
                            logf.write(f'  PPL(model1): NaN\n')
                            logf.write(f'  PPL(model2): NaN\n')
                            continue
                        elif np.isnan(ppl1):
                            avg = ppl2
                        elif np.isnan(ppl2):
                            avg = ppl1
                        else:
                            avg = (ppl1 + ppl2) / 2
                        ppl_scores.append((span_txt, avg))

                        logf.write(f'Span: "{span_txt}"\n')
                        logf.write(f'  PPL(model1): {ppl1:.3f}\n')
                        logf.write(f'  PPL(model2): {ppl2:.3f}\n')
                        logf.write(f'  Avg PPL    : {avg:.3f}\n')

    
                    if not ppl_scores:
                        logf.write(f"\n{'='*80}\n")
                        logf.write(f"⚠️  WARNING: Round {i+1} has no valid spans! Skipping this question...\n")
                        logf.write(f"{'='*80}\n")
                        logf.write(f"Debug Information:\n")
                        logf.write(f"  Current Prefix Length: {len(current_prefix)} chars\n")
                        logf.write(f"  Last 100 chars of prefix: ...{current_prefix[-100:]}\n\n")
                        logf.write(f"  Model1 generated {debug_info['model1_count']} candidate(s): {debug_info['model1_spans']}\n")
                        logf.write(f"  Model2 generated {debug_info['model2_count']} candidate(s): {debug_info['model2_spans']}\n\n")
                        logf.write(f"  Total unique spans collected: {len(cands)}\n")
                        logf.write(f"  Spans with both PPLs NaN (skipped): {nan_count}\n\n")
                        
                        if len(cands) == 0:
                            logf.write(f"  Root Cause: 🔴 Both models failed to generate any candidate spans.\n")
                            logf.write(f"              Possible reasons:\n")
                            logf.write(f"              - First word generation returned empty (first_word_ids is empty)\n")
                            logf.write(f"              - All generated text contains only punctuation\n")
                            logf.write(f"              - Generation config issue or model output problem\n")
                        elif nan_count == len(cands):
                            logf.write(f"  Root Cause: 🔴 All candidate spans have NaN PPL values from both models.\n")
                            logf.write(f"              This indicates potential issues with:\n")
                            logf.write(f"              - Token ID extraction or decoding\n")
                            logf.write(f"              - PPL computation for the given prefix\n")
                            logf.write(f"              - Model inference errors\n")
                        else:
                            logf.write(f"  Root Cause: 🔴 Unknown issue - some spans generated but none have valid PPL.\n")
                        
                        logf.write(f"{'='*80}\n\n")
                        
                        print(f"\n⚠️  [GPU {gpu_id}] Round {i+1} WARNING: No valid spans!")
                        print(f"   Model1: {debug_info['model1_count']} candidates, Model2: {debug_info['model2_count']} candidates")
                        print(f"   Total unique spans: {len(cands)}, NaN PPL count: {nan_count}")
                        if len(cands) == 0:
                            print(f"   Root cause: Both models failed to generate any candidate spans")
                        elif nan_count == len(cands):
                            print(f"   Root cause: All {len(cands)} span(s) have NaN PPL from both models")
                        print(f"   Check log file for details: {log_file}\n", flush=True)
                        break
                    selected_span = min(ppl_scores, key=lambda x: x[1])[0]

                    logf.write(f'Selected Span: "{selected_span}" with Avg PPL: '
                            f'{min(ppl_scores, key=lambda x: x[1])[1]:.3f}\n')

                if re.search(r'\\?boxed\s*\{', selected_span, flags=re.IGNORECASE):
                    kept_part = extract_boxed_answer(selected_span)
                    if kept_part:
                        current_prefix += " " + kept_part
                    break
                elif any(marker in selected_span.lower() for marker in ["question:", "note:"]) or "<" in selected_span:
                    kept_part = clean_text_before_question(selected_span)
                    if kept_part:
                        current_prefix += " " + kept_part
                    break
                else:
                    current_prefix += " " + selected_span


                span_trace.append({
                    "round": i + 1,
                    "candidates": [(s, round(p, 3)) for s, p in ppl_scores],
                    "selected": selected_span
                })


                tokenized_prefix = tokenizer1(current_prefix, return_tensors="pt")["input_ids"][0]


                if (tokenizer1.eos_token and tokenizer1.eos_token in selected_span) or \
                   (tokenizer2.eos_token and tokenizer2.eos_token in selected_span):
                    break

 
            if "Answer:" in current_prefix:
                answer_part = current_prefix.split("Answer:")[-1]
                answer_text = clean_text_before_question(answer_part.strip())
            else:
                answer_text = clean_text_before_question(current_prefix.replace(prompt, '', 1).strip())

            sample_num = batch_count * args.per_device_batch_size + idx + 1
            total_samples = end_idx - start_idx
            print(f"[GPU {gpu_id}] [{sample_num}/{total_samples}] Label: {answers[idx]} | Pred: {answer_text}", flush=True)

            sample_result = {
                "question": orig_question,
                "pred": answer_text,
                "label": answers[idx],

            }
            all_results.append(sample_result)
        
        batch_count += 1


    output_file = f"{args.output_file}_gpu{gpu_id}.json"
    with open(output_file, "w", encoding="utf-8") as fw:
        for sample in all_results:
            fw.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    total_model1 = forward_count_model1["generate"] + forward_count_model1["ppl"]
    total_model2 = forward_count_model2["generate"] + forward_count_model2["ppl"]
    total_all = total_model1 + total_model2
    
    print(f"\n{'='*70}")
    print(f"GPU {gpu_id} Forward Inference Statistics:")
    print(f"  Model1:")
    print(f"    Generate: {forward_count_model1['generate']:6d}")
    print(f"    PPL:      {forward_count_model1['ppl']:6d}")
    print(f"    Subtotal: {total_model1:6d}")
    print(f"  Model2:")
    print(f"    Generate: {forward_count_model2['generate']:6d}")
    print(f"    PPL:      {forward_count_model2['ppl']:6d}")
    print(f"    Subtotal: {total_model2:6d}")
    print(f"  Total:    {total_all:6d}")
    print(f"{'='*70}\n")
    
    print(f"GPU {gpu_id}: Completed processing, saved to {output_file}")


def merge_results(args, num_gpus):

    print("Merging results from all GPUs...")
    
    all_results = []
    temp_files = []
    

    for gpu_id in range(num_gpus):
        temp_file = f"{args.output_file}_gpu{gpu_id}.json"
        temp_files.append(temp_file)
        
        if os.path.exists(temp_file):
            with open(temp_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))
        else:
            print(f"Warning: {temp_file} not found!")
    
    
    with open(args.output_file, "w", encoding="utf-8") as fw:
        for sample in all_results:
            fw.write(json.dumps(sample, ensure_ascii=False) + "\n")
    

    print("Temporary GPU files are kept for safety:")
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            print(f"  Kept temporary file: {temp_file}")
    

    
    print(f"Merging completed. Final results saved to: {args.output_file}")
    print(f"Total samples processed: {len(all_results)}")



def main():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--test_set", type=str, help="")
    arg_parse.add_argument("--prompts", type=str, help="")
    arg_parse.add_argument("--model_path1", type=str, help="")
    arg_parse.add_argument("--model_path2", type=str, help="")
    arg_parse.add_argument("--output_file", type=str, help="")
    arg_parse.add_argument("--per_device_batch_size", type=int, default=1)
    arg_parse.add_argument("--max_new_tokens", type=int, default=10)
    arg_parse.add_argument("--span_length", type=int, default=4)
    arg_parse.add_argument("--max_span_rounds", type=int, default=5)
    arg_parse.add_argument("--beam_size", type=int, default=1, help="")
    arg_parse.add_argument("--voting_strategy", type=str, choices=["top1", "softmax"], default="top1",
                             help="")
    arg_parse.add_argument("--max_total_tokens", type=int, default=512, help="")
    arg_parse.add_argument("--theta_delta", type=float, default=0.7, help="Top-1 margin threshold")
    arg_parse.add_argument("--max_words", type=int, default=3, choices=[2, 3],
                             help="Max words to generate per step: 2 or 3 (default 3)")
    args = arg_parse.parse_args()

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")
    
    if num_gpus == 0:
        print("No GPUs detected! Please ensure CUDA is available.")
        return
    
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompt_complex = f.read()
    

    test_dataset = load_dataset("json", data_files=args.test_set, split='train')
    total_samples = len(test_dataset)
    print(f"Total samples: {total_samples}")
    
    samples_per_gpu = total_samples // num_gpus
    remainder = total_samples % num_gpus
    

    processes = []
    
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * samples_per_gpu
        if gpu_id == num_gpus - 1:  
            end_idx = total_samples
        else:
            end_idx = (gpu_id + 1) * samples_per_gpu
        
        print(f"GPU {gpu_id}: samples {start_idx} to {end_idx-1} (total: {end_idx-start_idx})")
        

        p = mp.Process(
            target=ensemble_decoding_worker,
            args=(gpu_id, start_idx, end_idx, args, prompt_complex)
        )
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    print("All GPU processes completed!")
    
    merge_results(args, num_gpus)
    

    print("Starting post-processing evaluation...")
    if 'gsm' in args.test_set.lower():
        gsm_parse_pred_ans(args.output_file)
    elif 'triviaqa' in args.test_set.lower() or 'nq' in args.test_set.lower():
        qa_parse_pred_ans(predictions_file=args.output_file, references_file=args.test_set, is_regex=False)
    elif 'arc' in args.test_set.lower() or 'piqa' in args.test_set.lower():
        arc_parse_pred_ans(args.output_file)
    
    print('Multi-GPU ensemble decoding completed!')


if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)
    main()
