import pandas as pd
import os
import re
import json
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv(override=True)
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file!")

# Azure OpenAI Configuration
RESOURCE_ENDPOINT = "https://gpt4o-dev-eschbach.openai.azure.com/"
DEPLOYMENT_NAME = "gpt-4o-mini"
API_VERSION = "2024-12-01-preview"

client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=RESOURCE_ENDPOINT,
    api_version=API_VERSION,
)

# ===============================
# COST TRACKING CONFIG
# ===============================
CHAT_INPUT_COST_PER_1K = 0.01  # $0.01 per 1k input tokens (adjust as per actual pricing)
CHAT_OUTPUT_COST_PER_1K = 0.1  # $0.10 per 1k output tokens (adjust as per actual pricing)

# Cost Counters
german_input_tokens = 0
german_output_tokens = 0
german_cost = 0.0

urdu_ident_input_tokens = 0
urdu_ident_output_tokens = 0
urdu_ident_cost = 0.0

urdu_trans_input_tokens = 0
urdu_trans_output_tokens = 0
urdu_trans_cost = 0.0


def extract_json(response_text: str):
    """Extract JSON object from GPT response reliably."""
    if not response_text:
        return None

    # Remove markdown code blocks if present
    response_text = re.sub(r"^```.*?\n", "", response_text, flags=re.S)
    response_text = re.sub(r"```$", "", response_text.strip())

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Fallback: Try to find the JSON structure using regex
    match = re.search(r"\{.*\}\s*$", response_text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    return None


# =========================================
# STEP 1: Generate German Translations
# =========================================
def generate_german_translations(idiom, meaning, german_idiom):
    """Create contextual English sentence + 3 German translations."""
    global german_input_tokens, german_output_tokens, german_cost

    prompt = f"""
    You are creating a translation evaluation dataset for English → German idioms.

    TASK:
    1️⃣ Create a natural English sentence using this idiom in bold:
    **{idiom}**

    2️⃣ Translate that sentence into German in 3 versions:
    A. Correct idiomatic translation - MUST use: "{german_idiom}"
    B. Alternative correct translation expressing the meaning but NOT idiomatic
    C. Literal incorrect translation (word-for-word, WRONG meaning)

    Return ONLY valid JSON with keys:
    english_sentence, target_A, target_B, target_C

    Idiom meaning: {meaning}
    """

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )

        usage = response.usage
        german_input_tokens += usage.prompt_tokens
        german_output_tokens += usage.completion_tokens
        german_cost += ((usage.prompt_tokens / 1000) * CHAT_INPUT_COST_PER_1K) + \
                       ((usage.completion_tokens / 1000) * CHAT_OUTPUT_COST_PER_1K)

        content = response.choices[0].message.content.strip()
        return extract_json(content)

    except Exception as e:
        print(f"Error in German gen: {e}")
        return None


# =========================================
# STEP 2: Identify Urdu Idiom Equivalents
# =========================================
def generate_urdu_equivalent(english_idiom, meaning, english_sentence):
    """Identify the closest Urdu idiom (Native & Roman)."""
    global urdu_ident_input_tokens, urdu_ident_output_tokens, urdu_ident_cost

    prompt = f"""
    You are an expert linguistic bridge between English and Urdu.

    Input:
    - English Idiom: "{english_idiom}"
    - Meaning: "{meaning}"
    - Context Sentence: "{english_sentence}"

    Task:
    Identify the closest Urdu idiom / proverb that conveys the same figurative meaning.

    Return ONLY JSON with keys:
    - urdu_idiom_native (Perso-Arabic script)
    - urdu_idiom_roman (Latin script)
    """

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )

        usage = response.usage
        urdu_ident_input_tokens += usage.prompt_tokens
        urdu_ident_output_tokens += usage.completion_tokens
        urdu_ident_cost += ((usage.prompt_tokens / 1000) * CHAT_INPUT_COST_PER_1K) + \
                           ((usage.completion_tokens / 1000) * CHAT_OUTPUT_COST_PER_1K)

        content = response.choices[0].message.content.strip()
        return extract_json(content)

    except Exception as e:
        print(f"Error in Urdu identification: {e}")
        return None


# =========================================
# STEP 3: Generate Comprehensive Urdu Translations
# =========================================
def generate_comprehensive_urdu_translations(english_sentence, idiom, meaning, urdu_native, urdu_roman):
    """
    Generates 6 Urdu translations: 3 in Native script (Perso-Arabic) and 3 in Roman script (Latin).
    """
    global urdu_trans_input_tokens, urdu_trans_output_tokens, urdu_trans_cost

    prompt = f"""
    You are an expert linguistic bridge creating an evaluation dataset for English → Urdu idioms.

    **Input Data:**
    - English Idiom: "{idiom}"
    - Meaning: "{meaning}"
    - English Sentence: "{english_sentence}"
    - Required Urdu Idiom (Native): "{urdu_native}"
    - Required Urdu Idiom (Roman): "{urdu_roman}"

    **TASK:**
    Translate the **English Sentence** into 6 distinct versions: 3 in Urdu Native script (Perso-Arabic) and 3 in Roman Urdu script (Latin).

    ### PART 1: URDU NATIVE SCRIPT (Perso-Arabic)
    A. **Native Idiomatic**:
       - MUST use the exact phrase "{urdu_native}".
       - Natural, grammatical Urdu.
    B. **Native Descriptive**:
       - Convey the exact meaning without the idiom.
       - Use standard descriptive Urdu.
    C. **Native Literal (Incorrect)**:
       - Translate the English idiom ("{idiom}") word-for-word literally.
       - It should sound awkward or wrong.

    ### PART 2: ROMAN URDU SCRIPT (Latin)
    D. **Roman Idiomatic**:
       - MUST use the exact phrase "{urdu_roman}".
       - Same sentence as A, but transliterated.
    E. **Roman Descriptive**:
       - Convey the exact meaning without the idiom.
       - Same sentence as B, but transliterated.
    F. **Roman Literal (Incorrect)**:
       - Translate the English idiom word-for-word literally.
       - Same sentence as C, but transliterated.

    **Return ONLY valid JSON with these 6 keys:**
    - "target_A_native"
    - "target_B_native"
    - "target_C_native"
    - "target_A_roman"
    - "target_B_roman"
    - "target_C_roman"
    """

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )

        usage = response.usage
        urdu_trans_input_tokens += usage.prompt_tokens
        urdu_trans_output_tokens += usage.completion_tokens
        urdu_trans_cost += ((usage.prompt_tokens / 1000) * CHAT_INPUT_COST_PER_1K) + \
                           ((usage.completion_tokens / 1000) * CHAT_OUTPUT_COST_PER_1K)

        content = response.choices[0].message.content.strip()
        return extract_json(content)

    except Exception as e:
        print(f"Error in Urdu Translation Gen: {e}")
        return None


# =========================================
# MAIN EXECUTION
# =========================================
def main():
    # Load input file
    input_file = "base.xlsx"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_excel(input_file)
    
    # Clean data
    df = df.dropna(subset=["Idiom", "Meaning(s)", "Matching German Idiom"])
    df["Idiom"] = df["Idiom"].str.strip()
    df = df[df["Idiom"] != ""].reset_index(drop=True)

    # For testing, you might want to limit rows:
    # df = df.head(3) 

    print(f"\nStarting pipeline on {len(df)} idioms...\n")

    full_dataset_rows = []

    for i, row in df.iterrows():
        idiom = row["Idiom"]
        meaning = row["Meaning(s)"]
        german_match = row["Matching German Idiom"]
        idiom_id = row.get("Idiom ID", i)

        print(f"[{i+1}/{len(df)}] Processing: {idiom}")

        # ----------------------------------------
        # STAGE 1: German Generation
        # ----------------------------------------
        german_out = None
        for attempt in range(3):
            german_out = generate_german_translations(idiom, meaning, german_match)
            if german_out: break
            time.sleep(1)
        
        if not german_out:
            print(f"   Failed German generation for: {idiom}")
            continue

        english_sentence = german_out["english_sentence"]

        # ----------------------------------------
        # STAGE 2: Urdu Identification
        # ----------------------------------------
        urdu_id_out = None
        for attempt in range(3):
            urdu_id_out = generate_urdu_equivalent(idiom, meaning, english_sentence)
            if urdu_id_out: break
            time.sleep(1)

        urdu_native = urdu_id_out.get("urdu_idiom_native") if urdu_id_out else None
        urdu_roman = urdu_id_out.get("urdu_idiom_roman") if urdu_id_out else None

        if not urdu_native or not urdu_roman:
            print(f"  Failed Urdu identification for: {idiom}")
            # We continue but Urdu columns will be empty
        
        # ----------------------------------------
        # STAGE 3: Comprehensive Urdu Translation
        # ----------------------------------------
        urdu_trans_out = None
        if urdu_native and urdu_roman:
            for attempt in range(3):
                urdu_trans_out = generate_comprehensive_urdu_translations(
                    english_sentence, idiom, meaning, urdu_native, urdu_roman
                )
                if urdu_trans_out: break
                time.sleep(1)
        
        if not urdu_trans_out:
             # Default empty dict if failed so .get() works
             urdu_trans_out = {}

        # ----------------------------------------
        # COLLECT DATA
        # ----------------------------------------
        full_dataset_rows.append({
            "Idiom ID": idiom_id,
            "English Idiom": idiom,
            "Meaning": meaning,
            "English Sentence": english_sentence,

            # German Data
            "German Idiom Match": german_match,
            "German Target A (Figurative)": german_out.get("target_A"),
            "German Target B (Alternative)": german_out.get("target_B"),
            "German Target C (Literal Incorrect)": german_out.get("target_C"),

            # Urdu Identification Data
            "Urdu Native Idiom": urdu_native,
            "Urdu Roman Idiom": urdu_roman,

            # Urdu Native Translations
            "Urdu Native Target A (Idiomatic)": urdu_trans_out.get("target_A_native"),
            "Urdu Native Target B (Descriptive)": urdu_trans_out.get("target_B_native"),
            "Urdu Native Target C (Literal)": urdu_trans_out.get("target_C_native"),

            # Urdu Roman Translations
            "Urdu Roman Target A (Idiomatic)": urdu_trans_out.get("target_A_roman"),
            "Urdu Roman Target B (Descriptive)": urdu_trans_out.get("target_B_roman"),
            "Urdu Roman Target C (Literal)": urdu_trans_out.get("target_C_roman"),
        })

    # ===============================
    # SAVE OUTPUT
    # ===============================
    final_df = pd.DataFrame(full_dataset_rows)
    output_filename = "idioms_complete_dataset.csv"
    final_df.to_csv(output_filename, index=False, encoding="utf-8-sig")

    print(f"\n Processing Complete! Saved to {output_filename}")

    # ===============================
    # COST SUMMARY
    # ===============================
    print("\n================ COST SUMMARY ================")
    
    print(f"German Generation Cost:")
    print(f"   - Input Tokens: {german_input_tokens}")
    print(f"   - Output Tokens: {german_output_tokens}")
    print(f"   - Cost: ${german_cost:.4f}")
    
    print(f"Urdu Identification Cost:")
    print(f"   - Input Tokens: {urdu_ident_input_tokens}")
    print(f"   - Output Tokens: {urdu_ident_output_tokens}")
    print(f"   - Cost: ${urdu_ident_cost:.4f}")

    print(f"Urdu Translation Cost:")
    print(f"   - Input Tokens: {urdu_trans_input_tokens}")
    print(f"   - Output Tokens: {urdu_trans_output_tokens}")
    print(f"   - Cost: ${urdu_trans_cost:.4f}")

    total_cost = german_cost + urdu_ident_cost + urdu_trans_cost
    print(f"\nFinal TOTAL COST: ${total_cost:.4f}")
    print("==============================================")


if __name__ == "__main__":
    main()