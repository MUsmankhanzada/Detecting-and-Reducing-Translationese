# Script to create Dataset which will be later used to evaluate the model and metrics for machine translation of idioms.
import pandas as pd
import os
import re
import json
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(override=True)
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file!")

RESOURCE_ENDPOINT = "https://gpt4o-dev-eschbach.openai.azure.com/"
DEPLOYMENT_NAME = "gpt-4o-mini"
API_VERSION = "2024-12-01-preview"

client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=RESOURCE_ENDPOINT,
    api_version=API_VERSION,
)

# ===============================
# COST TRACKING (SEPARATE)
# ===============================
CHAT_INPUT_COST_PER_1K = 0.01
CHAT_OUTPUT_COST_PER_1K = 0.1

# Separate counters
german_input_tokens = 0
german_output_tokens = 0
german_cost = 0.0

urdu_input_tokens = 0
urdu_output_tokens = 0
urdu_cost = 0.0


def extract_json(response_text: str):
    """Extract JSON object from GPT response reliably."""
    if not response_text:
        return None

    response_text = re.sub(r"^```.*?\n", "", response_text, flags=re.S)
    response_text = re.sub(r"```$", "", response_text.strip())

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}\s*$", response_text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    return None


# =========================================
# Generate German translations
# =========================================
def generate_translations(idiom, meaning, german_idiom):
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

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )

    usage = response.usage

    # Track GERMAN costs
    german_input_tokens += usage.prompt_tokens
    german_output_tokens += usage.completion_tokens

    german_cost += ((usage.prompt_tokens / 1000) * CHAT_INPUT_COST_PER_1K) + \
                   ((usage.completion_tokens / 1000) * CHAT_OUTPUT_COST_PER_1K)

    content = response.choices[0].message.content.strip()
    parsed = extract_json(content)

    if parsed is None:
        print(" JSON extraction failed — saving raw text for debugging")
        return None

    return parsed


# =========================================
# Generate Urdu idiom equivalents
# =========================================
def generate_urdu_equivalent(english_idiom, meaning, english_sentence):
    global urdu_input_tokens, urdu_output_tokens, urdu_cost

    prompt = f"""
You are an expert linguistic bridge between English and Urdu.

Input:
- English Idiom: "{english_idiom}"
- Meaning: "{meaning}"
- Context Sentence: "{english_sentence}"

Task:
Identify the closest Urdu idiom / proverb that conveys the same figurative meaning.

Return ONLY JSON:
- urdu_idiom_native
- urdu_idiom_roman
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )

    usage = response.usage

    # Track URDU costs
    urdu_input_tokens += usage.prompt_tokens
    urdu_output_tokens += usage.completion_tokens

    urdu_cost += ((usage.prompt_tokens / 1000) * CHAT_INPUT_COST_PER_1K) + \
                 ((usage.completion_tokens / 1000) * CHAT_OUTPUT_COST_PER_1K)

    content = response.choices[0].message.content.strip()
    parsed = extract_json(content)

    if parsed is None:
        print(" JSON extraction failed — saving raw text for debugging")
        return None

    return parsed


# =========================================
# MAIN
# =========================================
def main():
    df = pd.read_excel("base.xlsx")

    df = df.dropna(subset=["Idiom", "Meaning(s)", "Matching German Idiom"])
    df["Idiom"] = df["Idiom"].str.strip()
    df = df[df["Idiom"] != ""].reset_index(drop=True)

    print(f"\n Running test on {len(df)} idioms\n")

    german_rows = []

    # ==============================
    # 1. German generation loop
    # ==============================
    df = df.head(3) # TESTING LIMIT
    for i, row in df.iterrows():
        print(f" Processing German #{i+1}: {row['Idiom']}")

        for attempt in range(3):
            try:
                output = generate_translations(
                    row["Idiom"],
                    row["Meaning(s)"],
                    row["Matching German Idiom"]
                )
                if output:
                    german_rows.append({
                        "Idiom ID": row["Idiom ID"],
                        "Idiom": row["Idiom"],
                        "Meaning": row["Meaning(s)"],
                        "German_Idiom": row["Matching German Idiom"],
                        "English Sentence": output["english_sentence"],
                        "Target A (Figurative)": output["target_A"],
                        "Target B (Alternative)": output["target_B"],
                        "Target C (Literal Incorrect)": output["target_C"],
                    })
                    break
            except Exception as e:
                print(f" Retry #{attempt+1} due to: {e}")
                time.sleep(1)

    out_df = pd.DataFrame(german_rows)

    # ============================================
    # 2. Urdu generation loop (append to same rows)
    # ============================================
    urdu_native_list = []
    urdu_roman_list = []

    for i, row in out_df.iterrows():
        print(f" Processing Urdu #{i+1}: {row['Idiom']}")

        output = None
        for attempt in range(3):
            try:
                output = generate_urdu_equivalent(
                    row["Idiom"],
                    row["Meaning"],
                    row["English Sentence"]
                )
                if output:
                    break
            except Exception as e:
                print(f" Retry #{attempt+1} due to: {e}")
                time.sleep(1)

        urdu_native_list.append(output.get("urdu_idiom_native") if output else None)
        urdu_roman_list.append(output.get("urdu_idiom_roman") if output else None)

    out_df["Urdu_Idiom_Native"] = urdu_native_list
    out_df["Urdu_Idiom_Roman"] = urdu_roman_list

    # Save final dataset
    out_df.to_csv("idioms_test_batch.csv", index=False, encoding="utf-8-sig")

    print("\n Dataset saved → idioms_test_batch.csv")

    # ===============================
    # PRINT COST SUMMARY
    # ===============================
    print("\n===== COST SUMMARY =====")
    print(f"German prompt tokens: {german_input_tokens}")
    print(f"German completion tokens: {german_output_tokens}")
    print(f"German total cost: ${german_cost:.4f}\n")

    print(f"Urdu prompt tokens: {urdu_input_tokens}")
    print(f"Urdu completion tokens: {urdu_output_tokens}")
    print(f"Urdu total cost: ${urdu_cost:.4f}\n")

    total_cost = german_cost + urdu_cost
    print(f"TOTAL COST (German + Urdu): ${total_cost:.4f}")


if __name__ == "__main__":
    main()
