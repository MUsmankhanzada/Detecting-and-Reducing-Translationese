import os
import pandas as pd
import numpy as np
import json
import subprocess
import argparse
import sys

# ── Configuration ──
# We use the Hybrid XL model (Balanced performance/memory)
MODEL_NAME = "google/metricx-24-hybrid-xl-v2p6"
BATCH_SIZE = 1  
MAX_LEN    = 1024

# ── Language Configs with ABSOLUTE PATHS ──
# Update these paths if your files move!
BASE_DIR = "/home/users/u103357/Detecting-and-Reducing-Translationese"

LANG_CONFIGS = {
    "de": {
        "filename": os.path.join(BASE_DIR, "idioms_test_batch.csv"),
        "col_src": "English Sentence",
        "col_a": ["Target A (Figurative)", "Target A"],
        "col_b": ["Target B (Alternative)", "Target B"],
        "col_c": ["Target C (Literal Incorrect)", "Target C"],
        "suffix": "german"
    },
    "ur-native": {
        "filename": os.path.join(BASE_DIR, "urdu_idioms_cleaned.csv"),
        "col_src": "English Sentence",
        "col_a": ["Native Idiomatic", "Target A (Native Idiomatic)"],
        "col_b": ["Native Descriptive", "Target B (Native Descriptive)"],
        "col_c": ["Native Literal", "Target C (Native Literal)"],
        "suffix": "urdu_native"
    },
    "ur-roman": {
        "filename": os.path.join(BASE_DIR, "urdu_idioms_cleaned.csv"),
        "col_src": "English Sentence",
        "col_a": ["Roman Idiomatic", "Target A (Roman Idiomatic)"],
        "col_b": ["Roman Descriptive", "Target B (Roman Descriptive)"],
        "col_c": ["Roman Literal", "Target C (Roman Literal)"],
        "suffix": "urdu_roman"
    }
}

def find_col(columns, candidates):
    """Helper to find the correct column name from a list of possibilities."""
    for cand in candidates:
        if cand in columns: return cand
        for col in columns:
            if cand in col: return col
    return None

def create_input_jsonl(src_list, hyp_list, ref_list, output_path):
    """Creates the JSONL input file required by MetricX."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for s, h, r in zip(src_list, hyp_list, ref_list):
            # For QE mode, reference is empty string
            r_val = r if r is not None else ""
            item = {"source": s, "hypothesis": h, "reference": r_val}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def parse_output_jsonl(file_path):
    """Reads scores from MetricX output file."""
    scores = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                scores.append(float(data["prediction"]))
    except Exception as e:
        print(f"Error parsing output: {e}")
        return []
    return scores

def run_inference(input_json, output_json, is_qe):
    """Calls the official MetricX prediction script via subprocess."""
    cmd = [
        sys.executable, "-m", "metricx24.predict",
        "--tokenizer", "google/mt5-xl",
        "--model_name_or_path", MODEL_NAME,
        "--max_input_length", str(MAX_LEN),
        "--batch_size", str(BATCH_SIZE),
        "--input_file", input_json,
        "--output_file", output_json
    ]
    if is_qe:
        cmd.append("--qe")
        
    print(f"   Running MetricX... (QE={is_qe})")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, choices=["de", "ur-native", "ur-roman"])
    args = parser.parse_args()
    
    config = LANG_CONFIGS[args.lang]
    print(f"=== MetricX-24 Evaluation: {args.lang} ===")

    # 1. Load Data
    csv_path = config["filename"]
    print(f"Loading data from: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"CRITICAL ERROR: File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    col_src = find_col(df.columns, [config["col_src"]])
    col_a = find_col(df.columns, config["col_a"])
    col_b = find_col(df.columns, config["col_b"])
    col_c = find_col(df.columns, config["col_c"])
    
    if not all([col_src, col_a, col_b, col_c]):
        print("Error: Missing one or more columns in the CSV.")
        return

    # Drop rows where any target is missing
    df = df.dropna(subset=[col_src, col_a, col_b, col_c])
    srcs = df[col_src].tolist()
    
    # 2. Define Tasks (QE = No Reference, Ref = With Reference)
    # Adding ALL permutations
    tasks = [
        # QE Mode (No Reference)
        ("Score_QE_A", col_a, None),
        ("Score_QE_B", col_b, None),
        ("Score_QE_C", col_c, None),
        
        # Reference Mode: Using A as Ref
        ("Score_B_Ref_A", col_b, col_a), 
        ("Score_C_Ref_A", col_c, col_a), 
        
        # Reference Mode: Using B as Ref
        ("Score_A_Ref_B", col_a, col_b), 
        ("Score_C_Ref_B", col_c, col_b),
        
        # Reference Mode: Using C as Ref
        ("Score_A_Ref_C", col_a, col_c), 
        ("Score_B_Ref_C", col_b, col_c)
    ]
    
    temp_in = "temp_input.jsonl"
    temp_out = "temp_output.jsonl"
    results = df.copy()

    # 3. Execution Loop
    for label, hyp_col, ref_col in tasks:
        print(f"\n▼ Processing: {label} ...")
        hyps = df[hyp_col].tolist()
        refs = df[ref_col].tolist() if ref_col else [None] * len(hyps)
        
        # A. Create Input File
        create_input_jsonl(srcs, hyps, refs, temp_in)
        
        # B. Run Model
        try:
            run_inference(temp_in, temp_out, is_qe=(ref_col is None))
            
            # C. Parse Scores
            scores = parse_output_jsonl(temp_out)
            if len(scores) == len(df):
                results[label] = scores
            else:
                print(f"Warning: Score count mismatch for {label}.")
        except Exception as e:
            print(f"Failed to run {label}: {e}")

    # 4. Calculate Statistics (Win Rates & Averages)
    # Note: MetricX is LOWER IS BETTER. 
    # A Win means Score A < Score B.
    
    total = len(results)
    
    # Win Rates based on QE scores
    win_ab = np.mean(results["Score_QE_A"] < results["Score_QE_B"])
    win_ac = np.mean(results["Score_QE_A"] < results["Score_QE_C"])
    win_bc = np.mean(results["Score_QE_B"] < results["Score_QE_C"])

    # Averages for ALL columns
    averages = {}
    for label, _, _ in tasks:
        if label in results.columns:
            averages[label] = float(results[label].mean())

    report = {
        "language": args.lang,
        "note": "MetricX scores are Lower-is-Better",
        "win_rates": {
            "A_beats_B": win_ab,
            "A_beats_C": win_ac,
            "B_beats_C": win_bc
        },
        "averages": averages
    }

    # 5. Save Final Results
    # Excel
    output_excel = f"metricx_results_{config['suffix']}.xlsx"
    results.to_excel(output_excel, index=False)
    
    # JSON
    output_json = f"report_metricx_{config['suffix']}.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)

    print(f"\nDone!")
    print(f" Excel: {output_excel}")
    print(f" JSON:  {output_json}")
    
    # Cleanup temp files
    if os.path.exists(temp_in): os.remove(temp_in)
    if os.path.exists(temp_out): os.remove(temp_out)

if __name__ == "__main__":
    main()