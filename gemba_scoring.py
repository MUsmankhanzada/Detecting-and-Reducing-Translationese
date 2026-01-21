#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import json
import argparse
import sys

# ── Import GEMBA ──
try:
    from gemba.utils import get_gemba_scores
    from gemba.gpt_api import GptApi
except ImportError:
    print("CRITICAL ERROR: Could not import GEMBA. Make sure you installed it with 'pip install -e .'")
    sys.exit(1)

# ── Configuration ──
DEFAULT_MODEL_NAME = "gpt-4" 
DEFAULT_METHOD = "GEMBA-DA" 

# ── Language Config Definitions ──
LANG_CONFIGS = {
    "de": {
        "filename": "idioms_test_batch.csv",
        "src_lang": "en",
        "tgt_lang": "de",
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
        "src_lang": "en",
        "tgt_lang": "ur",
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
        "src_lang": "en",
        "tgt_lang": "ur_roman", 
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
    for cand in candidates:
        if cand in columns:
            return cand
        for col in columns:
            if cand in col:
                return col
    return None

def load_data(filepath, config):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    cols = df.columns.tolist()
    
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

    df = df.dropna(subset=[src_col, col_a, col_b, col_c])

    return {
        'src': df[src_col].tolist(),
        'cand_a': df[col_a].tolist(),
        'cand_b': df[col_b].tolist(),
        'cand_c': df[col_c].tolist(),
        'df_original': df  
    }

def run_gemba_batch(sources, hypotheses, src_lang, tgt_lang, method, model_name, label):
    """
    Wrapper to call GEMBA scoring logic
    """
    print(f"   Scoring {len(sources)} items for {label} (Method: {method})...")
    
    try:
        # --- FIX: USE POSITIONAL ARGUMENTS ---
        # Supervisor's script used: (sources, hypotheses, src_lang, tgt_lang, method, model)
        scores = get_gemba_scores(
            sources,      # 1. Source List
            hypotheses,   # 2. Target List
            src_lang,     # 3. Source Lang Code
            tgt_lang,     # 4. Target Lang Code
            method,       # 5. Method (GEMBA-DA)
            model_name    # 6. Model (gpt-4)
        )
        return scores
    except Exception as e:
        print(f"CRITICAL ERROR during GEMBA scoring: {e}")
        # Return zeros on error so script doesn't crash entirely
        return [0.0] * len(sources)

def main():
    parser = argparse.ArgumentParser(description="Run GEMBA scoring for idioms.")
    parser.add_argument("--lang", type=str, required=True, choices=["de", "ur-native", "ur-roman"],
                        help="Choose language: 'de', 'ur-native', 'ur-roman'")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, 
                        help=f"Azure Deployment Name (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, 
                        help=f"Scoring method (default: {DEFAULT_METHOD})")
    parser.add_argument("--input", type=str, default=None, help="Override input CSV")
    parser.add_argument("--output_dir", type=str, default="gemba_results", help="Directory to save results")
    
    args = parser.parse_args()
    config = LANG_CONFIGS[args.lang]
    input_file = args.input if args.input else config["filename"]
    
    setup_directories(args.output_dir)
    
    # 1. Load Data
    data = load_data(input_file, config)
    if not data: return
    src = data['src']
    
    print(f"\n=== Starting GEMBA Run ===")
    print(f"Model:  {args.model}")
    print(f"Method: {args.method}")
    print(f"Pairs:  {config['src_lang']} -> {config['tgt_lang']}")
    
    # 2. Run Scoring (3 Passes)
    scores = {}
    
    # Score Target A
    scores['A'] = run_gemba_batch(src, data['cand_a'], 
                                  config['src_lang'], config['tgt_lang'], 
                                  args.method, args.model, f"A ({config['label_a']})")
    
    # Score Target B
    scores['B'] = run_gemba_batch(src, data['cand_b'], 
                                  config['src_lang'], config['tgt_lang'], 
                                  args.method, args.model, f"B ({config['label_b']})")
    
    # Score Target C
    scores['C'] = run_gemba_batch(src, data['cand_c'], 
                                  config['src_lang'], config['tgt_lang'], 
                                  args.method, args.model, f"C ({config['label_c']})")

    # 3. Calculate Win Rate & Averages
    # (Check if scores are not empty lists of zeros)
    if not scores['A'] or scores['A'][0] == 0.0:
        print("\nWARNING: Scores appear to be empty or failed. Checking data...")

    win_rate = sum(1 for a, b in zip(scores['A'], scores['B']) if a > b) / len(src)
    
    print("\n=== Results Summary ===")
    print(f"Average A ({config['label_a']}): {np.mean(scores['A']):.2f}")
    print(f"Average B ({config['label_b']}): {np.mean(scores['B']):.2f}")
    print(f"Average C ({config['label_c']}): {np.mean(scores['C']):.2f}")
    print(f"Win Rate (A > B): {win_rate:.2%}")

    # 4. Save Results
    suffix = config['result_suffix']
    
    # JSON
    final_report = {
        'language': args.lang,
        'model': args.model,
        'method': args.method,
        'averages': {
            f"A_{config['label_a']}": np.mean(scores['A']),
            f"B_{config['label_b']}": np.mean(scores['B']),
            f"C_{config['label_c']}": np.mean(scores['C'])
        },
        'win_rate_A_vs_B': win_rate
    }
    json_path = os.path.join(args.output_dir, f"gemba_report_{suffix}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    # Excel
    df_out = data['df_original'].copy()
    df_out[f'Score_A_{config["label_a"]}'] = scores['A']
    df_out[f'Score_B_{config["label_b"]}'] = scores['B']
    df_out[f'Score_C_{config["label_c"]}'] = scores['C']
    
    excel_path = os.path.join(args.output_dir, f"gemba_results_{suffix}.xlsx")
    df_out.to_excel(excel_path, index=False)
    
    print(f"\nSaved results to: {excel_path}")

if __name__ == "__main__":
    main()