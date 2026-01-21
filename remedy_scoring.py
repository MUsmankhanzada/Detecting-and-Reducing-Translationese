#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import json
import argparse
import sys
from transformers import AutoTokenizer

# ── Import ReMedy directly ──
try:
    from remedy.toolbox.score import (
        initialize_model, prepare_dataset,
        process_data_for_scoring, calculate_scores
    )
except ImportError:
    # Fallback: Try to add local folder if installed there
    sys.path.append(os.path.join(os.getcwd(), "Remedy"))
    from remedy.toolbox.score import (
        initialize_model, prepare_dataset,
        process_data_for_scoring, calculate_scores
    )

# ── Configuration Constants ──
MODEL_ID    = "/project/home/p201026/Models/ReMedy-9B-23"
MAX_LENGTH  = 4096
NUM_GPUS    = 1 
CACHE_DIR   = None

# ── Language Config Definitions ──
LANG_CONFIGS = {
    "de": {
        "filename": "idioms_test_batch.csv",
        "tgt_lang_code": "de",
        "result_suffix": "german",
        "col_src": "English Sentence",
        "col_a": ["Target A (Figurative)", "Target A"],
        "col_b": ["Target B (Alternative)", "Target B"],
        "col_c": ["Target C (Literal Incorrect)", "Target C"],
        "label_a": "Figurative",
        "label_b": "Literal",
        "label_c": "Incorrect"
    },
    "ur-native": {
        "filename": "urdu_idioms_cleaned.csv",
        "tgt_lang_code": "ur",
        "result_suffix": "urdu_native",
        "col_src": "English Sentence",
        "col_a": ["Native Idiomatic", "Target A (Native Idiomatic)"],
        "col_b": ["Native Descriptive", "Target B (Native Descriptive)"],
        "col_c": ["Native Literal", "Target C (Native Literal)"],
        "label_a": "Native Idiomatic",
        "label_b": "Native Descriptive",
        "label_c": "Native Literal"
    },
    "ur-roman": {
        "filename": "urdu_idioms_cleaned.csv",
        "tgt_lang_code": "ur_rom", 
        "result_suffix": "urdu_roman",
        "col_src": "English Sentence",
        "col_a": ["Roman Idiomatic", "Target A (Roman Idiomatic)"],
        "col_b": ["Roman Descriptive", "Target B (Roman Descriptive)"],
        "col_c": ["Roman Literal", "Target C (Roman Literal)"],
        "label_a": "Roman Idiomatic",
        "label_b": "Roman Descriptive",
        "label_c": "Roman Literal"
    }
}

def setup_directories(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def find_col(columns, candidates):
    """Helper to find a column name from a list of possible candidates"""
    for cand in candidates:
        if cand in columns:
            return cand
        for col in columns:
            if cand in col:
                return col
    return None

def load_data(filepath, config):
    print(f"▼ Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    cols = df.columns.tolist()
    
    # Identify Columns based on Config
    src_col = find_col(cols, [config["col_src"]])
    col_a = find_col(cols, config["col_a"])
    col_b = find_col(cols, config["col_b"])
    col_c = find_col(cols, config["col_c"])

    print(f"  [{config['result_suffix']}] Columns Found:")
    print(f"   - Src: {src_col}")
    print(f"   - A:   {col_a}")
    print(f"   - B:   {col_b}")
    print(f"   - C:   {col_c}")

    if not all([src_col, col_a, col_b, col_c]):
        print(f"CRITICAL ERROR: Missing columns for language mode.")
        return None

    # Filter out empty rows
    df = df.dropna(subset=[src_col, col_a, col_b, col_c])

    return {
        'src': df[src_col].tolist(),
        'cand_a': df[col_a].tolist(),
        'cand_b': df[col_b].tolist(),
        'cand_c': df[col_c].tolist(),
        'df_original': df  
    }

def get_remedy_scores(llm, tokenizer, srcs, tgts, tgt_lang, refs=None, task_name="task"):
    print(f"   ↳ Scoring {len(srcs)} items for {task_name}...")
    
    if refs is None:
        refs = [''] * len(srcs)
        is_qe = True
    else:
        is_qe = False

    df = pd.DataFrame({
        "src": srcs,
        "mt" : tgts,
        "ref": refs,
        "lp" : [f"en-{tgt_lang}"] * len(srcs),
        "seg_id": range(len(srcs)),
        "system-name": ['sys'] * len(srcs),
        "human_ratings": [0.0] * len(srcs)
    })

    # Processing
    ds, _ = process_data_for_scoring(df, QE=is_qe)
    ds    = prepare_dataset(ds, tokenizer, MAX_LENGTH, enable_truncate=True)
    
    # Fix for vLLM: convert to Python lists
    raw_inputs = ds["input_ids_chosen"]
    clean_inputs = [list(x) for x in raw_inputs]

    # Inference
    emb = llm.encode(prompt_token_ids=clean_inputs, use_tqdm=True)
    scored = calculate_scores(emb, df, calibrate=True)
    
    return scored["sigmoid:seg"].tolist()

def main():
    # ── 1. Argument Parsing ──
    parser = argparse.ArgumentParser(description="Run ReMedy scoring for idioms.")
    parser.add_argument("--lang", type=str, required=True, choices=["de", "ur-native", "ur-roman"],
                        help="Choose language to set defaults.")
    parser.add_argument("--input", type=str, default=None, help="Override input CSV file")
    parser.add_argument("--output_dir", type=str, default="testcase_results", help="Directory to save results")
    
    args = parser.parse_args()
    config = LANG_CONFIGS[args.lang]
    
    # Determine which file to use
    input_file = args.input if args.input else config["filename"]
    
    print(f"=== Starting Run: {args.lang} ===")
    print(f"=== Input File:   {input_file} ===")
    setup_directories(args.output_dir)

    # ── 2. Load Data ──
    data = load_data(input_file, config)
    if not data: return
    src = data['src']
    
    # ── 3. Initialize Model ──
    print(f"\n▼ Initializing ReMedy on GPU ({NUM_GPUS} GPU)...")
    try:
        llm = initialize_model(
            MODEL_ID, MAX_LENGTH, enable_truncate=True, 
            num_gpus=NUM_GPUS, num_seqs=256, cache_dir=CACHE_DIR,
            gpu_memory_utilization=0.90
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model. {e}")
        return

    # ── 4. Scoring (QE Mode) ──
    scores_qe = {}
    tgt_code = config["tgt_lang_code"]
    
    print(f"\n--- QE Mode (en-{tgt_code}) ---")
    scores_qe['A'] = get_remedy_scores(llm, tokenizer, src, data['cand_a'], tgt_code, None, f"A ({config['label_a']})")
    scores_qe['B'] = get_remedy_scores(llm, tokenizer, src, data['cand_b'], tgt_code, None, f"B ({config['label_b']})")
    scores_qe['C'] = get_remedy_scores(llm, tokenizer, src, data['cand_c'], tgt_code, None, f"C ({config['label_c']})")

    # ── 5. Reference Experiments ──
    print("\n--- Reference Mode ---")
    
    # Using Reference A
    s_b_ref_a = get_remedy_scores(llm, tokenizer, src, tgts=data['cand_b'], tgt_lang=tgt_code, refs=data['cand_a'], task_name="Tgt B | Ref A")
    s_c_ref_a = get_remedy_scores(llm, tokenizer, src, tgts=data['cand_c'], tgt_lang=tgt_code, refs=data['cand_a'], task_name="Tgt C | Ref A")

    # Using Reference B
    s_a_ref_b = get_remedy_scores(llm, tokenizer, src, tgts=data['cand_a'], tgt_lang=tgt_code, refs=data['cand_b'], task_name="Tgt A | Ref B")
    s_c_ref_b = get_remedy_scores(llm, tokenizer, src, tgts=data['cand_c'], tgt_lang=tgt_code, refs=data['cand_b'], task_name="Tgt C | Ref B")

    # Using Reference C
    s_a_ref_c = get_remedy_scores(llm, tokenizer, src, tgts=data['cand_a'], tgt_lang=tgt_code, refs=data['cand_c'], task_name="Tgt A | Ref C")
    s_b_ref_c = get_remedy_scores(llm, tokenizer, src, tgts=data['cand_b'], tgt_lang=tgt_code, refs=data['cand_c'], task_name="Tgt B | Ref C")

    # ── 6. Save Results ──
    suffix = config['result_suffix']
    
    # Calculate Win Rates
    total_samples = len(src)
    win_rate_ab = sum(1 for a, b in zip(scores_qe['A'], scores_qe['B']) if a > b) / total_samples
    win_rate_ac = sum(1 for a, c in zip(scores_qe['A'], scores_qe['C']) if a > c) / total_samples
    win_rate_bc = sum(1 for b, c in zip(scores_qe['B'], scores_qe['C']) if b > c) / total_samples

    final_report = {
        'language': args.lang,
        'averages': {
            f"QE_A_{config['label_a']}": np.mean(scores_qe['A']),
            f"QE_B_{config['label_b']}": np.mean(scores_qe['B']),
            f"QE_C_{config['label_c']}": np.mean(scores_qe['C'])
        },
        'win_rates': {
            'A_vs_B': win_rate_ab,
            'A_vs_C': win_rate_ac,
            'B_vs_C': win_rate_bc
        }
    }
    
    json_path = os.path.join(args.output_dir, f"report_{suffix}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    # Excel Report
    df_out = data['df_original'].copy()
    
    # QE Scores
    df_out[f'Score_QE_A_{config["label_a"]}'] = scores_qe['A']
    df_out[f'Score_QE_B_{config["label_b"]}'] = scores_qe['B']
    df_out[f'Score_QE_C_{config["label_c"]}'] = scores_qe['C']
    
    # Reference Scores
    df_out['Score_B_ref_A'] = s_b_ref_a
    df_out['Score_C_ref_A'] = s_c_ref_a
    
    df_out['Score_A_ref_B'] = s_a_ref_b
    df_out['Score_C_ref_B'] = s_c_ref_b
    
    df_out['Score_A_ref_C'] = s_a_ref_c
    df_out['Score_B_ref_C'] = s_b_ref_c
    
    excel_path = os.path.join(args.output_dir, f"results_{suffix}.xlsx")
    df_out.to_excel(excel_path, index=False)
    
    print(f"\n✓ Done! Results saved to:\n  - {json_path}\n  - {excel_path}")

if __name__ == "__main__":
    main()