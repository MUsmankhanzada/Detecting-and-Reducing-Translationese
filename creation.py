# Script to create Dataset which will be later used to evaluate the model and metrics for machine translation of idioms.
import pandas as pd
import os
import re
import json
import time
from dotenv import load_dotenv
from openai import AzureOpenAI
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

# =========================
# Cost Tracking
# =========================
CHAT_INPUT_COST_PER_1K = 0.01
CHAT_OUTPUT_COST_PER_1K = 0.1

total_input_tokens = 0
total_output_tokens = 0
total_cost = 0.0


def extract_json(response_text: str):
    """Extract JSON object from GPT response reliably."""
    if not response_text:
        return None

    # Remove markdown code fences like ```json ... ```
    response_text = re.sub(r"^```.*?\n", "", response_text, flags=re.S)
    response_text = re.sub(r"```$", "", response_text.strip())

    # Try direct JSON load
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON inside
    match = re.search(r"\{.*\}\s*$", response_text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    return None



def generate_translations(idiom, meaning, german_idiom):
    """Create contextual English sentence + 3 German translations."""
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
    
    # print("\n----- RAW GPT OUTPUT -----")
    # print(response.choices[0].message.content)
    # print("-------------------------\n")

    usage = response.usage
    global total_input_tokens, total_output_tokens, total_cost

    total_input_tokens += usage.prompt_tokens
    total_output_tokens += usage.completion_tokens

    input_cost = (usage.prompt_tokens / 1000) * CHAT_INPUT_COST_PER_1K
    output_cost = (usage.completion_tokens / 1000) * CHAT_OUTPUT_COST_PER_1K
    total_cost += (input_cost + output_cost)

    content = response.choices[0].message.content.strip()
    parsed = extract_json(content)

    if parsed is None:
        print(" JSON extraction failed — saving raw text for debugging")
        return {"english_sentence": None, "target_A": None, "target_B": None, "target_C": None}

    return parsed



def main():
    df = pd.read_excel("base.xlsx")

    # Clean dataset
    df = df.dropna(subset=["Idiom", "Meaning(s)", "Matching German Idiom"])
    df["Idiom"] = df["Idiom"].str.strip()
    df = df[df["Idiom"] != ""].reset_index(drop=True)

    # #  TEST MODE (only 2 rows)
    # df = df.head(2)
    print(f" Running test on {len(df)} idioms\n")

    results = []
    for i, row in df.iterrows():
        print(f" Processing #{i+1}: {row['Idiom']}")

        for attempt in range(3):  # retries
            try:
                output = generate_translations(
                    row["Idiom"],
                    row["Meaning(s)"],
                    row["Matching German Idiom"]
                )
                if output:
                    results.append({
                        "Idiom ID": row["Idiom ID"],
                        "Idiom": row["Idiom"],
                        "Meaning": row["Meaning(s)"],
                        "English Sentence": output["english_sentence"],
                        "Target A (Figurative)": output["target_A"],
                        "Target B (Alternative)": output["target_B"],
                        "Target C (Literal Incorrect)": output["target_C"],
                    })
                    break
            except Exception as e:
                print(f" Retry #{attempt+1} due to: {e}")
                time.sleep(1)

    out_df = pd.DataFrame(results)
    out_df.to_csv("idioms_test_batch.csv", index=False, encoding="utf-8-sig")

    print("\n Test dataset saved to: idioms_test_batch.csv")

    # Cost summary
    print("\n Token Usage Summary")
    print(f"Prompt tokens used: {total_input_tokens}")
    print(f"Completion tokens used: {total_output_tokens}")
    print(f"Estimated total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
