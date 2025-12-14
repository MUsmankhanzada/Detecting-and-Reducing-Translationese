# SUBTASK: Multilingual Idiom Translation Evaluation Dataset  
**English → German, English → Urdu (Roman & Native Scripts)**

## Overview

This project focuses on creating a **high-quality evaluation dataset for machine translation of idioms**, covering:

- **English → German**
- **English → Urdu (Roman script)**
- **English → Urdu (Native Perso-Arabic script)**

The dataset is explicitly designed to test a model’s ability to **distinguish between**:

1. **Correct idiomatic translations**
2. **Correct non-idiomatic (descriptive) alternatives**
3. **Incorrect literal (word-for-word) translations**

Such distinctions are critical for evaluating semantic and cultural understanding in machine translation systems.

---

## Input Data

The project starts from a base dataset:

**`base.xlsx`** (300 entries), containing:
- English Idiom  
- English Meaning  
- Matching German Equivalent Idiom  

---

## Methodology

Dataset creation followed a **multi-stage pipeline** combining automated generation using Large Language Models (LLMs) and **rigorous human-in-the-loop curation by native speakers**.

---

## Step 1: German Dataset Generation

Using **`gpt-4o-mini`**, we generated an English context sentence and three German translations per idiom.

### Inputs
- English Idiom  
- English Meaning  
- German Equivalent Idiom  

### Outputs
- **English Context Sentence**  
  A natural sentence using the English idiom.

- **Target A (Idiomatic)**  
  Correct German translation using the provided idiomatic equivalent.

- **Target B (Alternative)**  
  Correct German translation conveying the meaning without using an idiom.

- **Target C (Literal)**  
  Incorrect, word-for-word translation of the English idiom.

### Validation
A **native German speaker** manually verified a subset of examples to ensure translation quality and idiomatic correctness.

---

## Step 2: Urdu Idiom Identification & Curation (Human-in-the-Loop)

This step required **significant manual intervention** due to the cultural and linguistic complexity of Urdu idioms.

### Initial Generation (GPT-4o-mini)
- Roman Urdu equivalents were first generated automatically.
- Results were reviewed by a native Urdu speaker and found to be **inconsistent**.

### Second Generation (GPT-5)
- The same task was re-run using **GPT-5**, yielding improved but still incomplete results.

### Third Option (Gemini)
- For difficult cases (≈ **50% of the dataset**), both GPT models failed to produce culturally accurate idioms.
- In these cases, **Gemini** was used to generate a third alternative.

### Manual Curation
A **native Urdu speaker manually reviewed all 300 idioms**, selecting the **best Roman Urdu equivalent** from:
- GPT-4o-mini
- GPT-5
- Gemini

This curated Roman Urdu idiom set serves as the **gold standard** for subsequent steps.

---

## Step 3: Urdu Script Conversion

Once Roman Urdu idioms were finalized, a model was used to **transliterate** them into **Native Urdu (Perso-Arabic script)**.

---

## Step 4: Urdu Translation Generation

Using the curated idioms, **GPT-5** was employed to generate the final translation dataset.

### Inputs
- English Idiom  
- English Meaning  
- English Context Sentence  
- Curated Roman Urdu Idiom  
- Native Urdu Idiom  

### Outputs (6 translations per entry)

**Native Urdu (Perso-Arabic):**
- Target A: Idiomatic  
- Target B: Descriptive (non-idiomatic)  
- Target C: Literal (incorrect)

**Roman Urdu (Latin):**
- Target A: Idiomatic  
- Target B: Descriptive (non-idiomatic)  
- Target C: Literal (incorrect)

---

## Scripts and Files

### `base.xlsx`
The initial dataset containing English idioms and their German equivalents.

### `datasetcreation.py`
A consolidated **one-shot script** that automates the generation logic for all stages described above.

> ⚠️ **Important Note**  
> Although this script contains the full pipeline logic, the **actual dataset was not created in a single automatic run**.  
> The Urdu Idiom Identification step relied on **iterative manual curation** using private notebooks.  
> Running this script end-to-end without that manual curation step will produce **lower-quality Urdu idiomatic equivalents**.

### `test.ipynb` *(Not Shared)*
A private Jupyter notebook used for:
- Iterative testing
- Model comparison (GPT-4o-mini vs GPT-5 vs Gemini)
- Manual selection and curation of Urdu idioms

---

## Quality Assurance

- **German:**  
  Verified by a native German speaker (some examples).

- **Urdu:**  
  Fully curated and verified by a native Urdu speaker across all **300 examples**, using a **multi-model approach (GPT + Gemini)** to ensure high-quality idiomatic equivalence.

---

## Intended Use

This dataset is intended for:
- Evaluation of machine translation systems
- Idiom translation benchmarks
- Analysis of literal vs idiomatic translation errors
- Cross-lingual and script-aware MT research

---

## Disclaimer

This dataset emphasizes **quality over full automation**.  
Human linguistic expertise was essential, particularly for Urdu idioms, where cultural nuance cannot be reliably captured by a single model.

