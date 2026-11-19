---
title: "DS-GA 1011 Assignment 4: Fine-tuning Language Models"
author: "Maanas Lalwani"
geometry: margin=1in
fontsize: 11pt
---

## Links

- **GitHub repository (Part 1 and Part 2 code):** https://github.com/ml1608/nyu-dsga1011-a4
- **Google Drive folder (model checkpoint used for Q7 outputs):** https://drive.google.com/file/d/1MnF_eH-GIILkVukFMvRimaCLMM-Eul2v/view?usp=drive_link

# Part 1: Fine-tuning BERT for Sentiment Classification

## Q1. Fine-tune BERT (10 points)

<!-- No written answer required; graded from out_original.txt. Optional one-line summary below. -->

Fine-tuned `bert-base-cased` on the full IMDB training set (3 epochs, lr 5e-5, batch size 8).
Test accuracy: **93.12%**.

## Q2a. Design a data transformation (10 points)

I designed a data transformation that combines **WordNet-based synonym replacement and QWERTY keyboard typos** to simulate variations in how users write movie reviews. The goal is to create realistic out-of-distribution (OOD) examples while generally preserving the original sentiment labels.

**Transformation process:**

1. **Tokenization:** Each review is split into tokens using NLTK's `word_tokenize()`. Only alphabetic words longer than three characters are eligible for modification.

2. **Synonym replacement (probability = 0.25):** For each eligible word, the transformation randomly decides whether to replace it with a synonym. Synonyms are collected from the lemmas of all WordNet synsets associated with that word. The original word is excluded, underscores in multi-word synonyms are replaced with spaces, and one candidate is selected randomly. If no alternative exists, the word remains unchanged.

3. **Keyboard typo introduction (probability = 0.30):** After synonym replacement, the transformation independently attempts to introduce a typo. It randomly selects exactly one character from those supported by a predefined QWERTY-neighbor mapping (`a, e, i, o, u, s, t, n, r, l`) and replaces it with a neighboring keyboard character. For example, `e` can become `r`, and `u` can become `i`. If the resulting synonym contains multiple words, the typo step is skipped.

4. **Text reconstruction:** The transformation preserves the original word's capitalization where applicable and reconstructs the review using NLTK's `TreebankWordDetokenizer`. I use `random.seed(0)` for reproducibility, although individual transformations depend on the order in which examples are processed.

**Illustrative example (negative sentiment):**

- **Original:** "This film is bad. It's filled with glaring plot holes."
- **Transformed:** "This fiom is bad . It's filled with glsring patch holes."

Here, `film` becomes `fiom` through a keyboard typo, while `plot` becomes `patch` through synonym replacement. Despite these changes, the negative sentiment remains understandable.

**Why this transformation is reasonable:**

Real users may express the same opinion using different vocabulary or make typing mistakes, particularly when writing quickly on a phone. Restricting modifications to words longer than three characters helps preserve short but important sentiment expressions such as "not" and "bad." Changing only one character per selected word also helps keep misspelled words recognizable.

One limitation is that WordNet does not account for the context or part of speech of a word, so some replacements may sound unnatural or occasionally change its meaning. The relatively high typo probability also makes this an aggressive robustness test rather than a perfect simulation of everyday writing.

Overall, this transformation allows me to investigate whether fine-tuned BERT genuinely recognizes sentiment across different word choices and spelling variations or relies too heavily on specific vocabulary and patterns encountered during training.

## Q2b. Implement the transformation (15 points)

<!-- No written answer required; graded from out_transformed.txt. Optional summary below. -->

| Model | Original test | Transformed test | Drop |
|---|---|---|---|
| BERT fine-tuned on original data | 93.12% | 87.49% | 5.63 points |

## Q3. Data augmentation (15 points)

| Model | Original test | Transformed test |
|---|---|---|
| Trained on original data (Q1) | 93.12% | 87.49% |
| Trained on original + 5,000 transformed examples (Q3) | 93.06% | 88.77% |

**1. Did performance on the transformed test data improve after augmentation?**

Yes. Accuracy on the transformed test set increased from **87.49% to 88.77%**, an improvement of **1.28 percentage points**. This suggests that training with augmented examples helped BERT become more robust to synonym replacements and spelling mistakes.

**2. How did augmentation affect performance on the original test data?**

Accuracy on the original test set decreased slightly from **93.12% to 93.06%**, a drop of just **0.06 percentage points**. This difference is very small and could be due to normal training variation. Overall, augmentation improved robustness without meaningfully sacrificing performance on the original data.

**3. Give an intuitive explanation for the observed results.**

The original model was trained only on clean reviews, whereas the augmented model also encountered reviews containing WordNet synonyms and realistic keyboard typos. This helped it learn to recognize sentiment despite changes in vocabulary and spelling. Since 25,000 of the 30,000 training examples were still original reviews, the model retained its ability to classify clean text accurately.

**4. Explain one limitation of this data augmentation approach.**

One limitation is that the model is trained on the same types of transformations used during evaluation. Although this improves performance on synonym replacements and keyboard typos, it may not generalize to other distribution shifts, such as slang, sarcasm, or different writing styles. Therefore, better performance on this transformed test set does not guarantee robustness to all out-of-distribution data.

# Part 2: Fine-tuning T5 for Text-to-SQL

## Q4. Data statistics and processing (5 points)

<!-- Use the T5 tokenizer (google-t5/t5-small) to compute these. -->

**Table 1: Data statistics before any pre-processing.**

| Statistics Name | Train | Dev |
|---|---|---|
| Number of examples | 4225 | 466 |
| Mean sentence length | 18.10 | 18.07 |
| Mean SQL query length | 217.37 | 211.05 |
| Vocabulary size (natural language) | 792 | 466 |
| Vocabulary size (SQL) | 556 | 396 |

**Table 2: Data statistics after pre-processing.**

| Statistics Name | Train | Dev |
|---|---|---|
| **Model name** | google-t5/t5-small | |
| Mean sentence length | 18.10 | 18.07 |
| Mean SQL query length | 217.25 | 210.94 |
| Vocabulary size (natural language) | 792 | 466 |
| Vocabulary size (SQL) | 556 | 396 |

For preprocessing, I used the `google-t5/t5-small` tokenizer for both natural-language inputs and SQL outputs. I replaced the `<` character in SQL queries with the special token `<extra_id_1>` because the original character was not represented correctly by the tokenizer. This allows the model to learn and generate comparison operators more reliably. The natural-language inputs were left unchanged.

Mean sequence lengths were calculated by averaging the number of T5 tokens per example, including the end-of-sequence token `</s>`. Vocabulary sizes represent the number of distinct token IDs observed in each dataset, rather than the tokenizer's full vocabulary size.

## Q5. T5 fine-tuning details (10 points)

**Table 3: Details of the best-performing T5 model configuration (fine-tuned).**

| Design choice | Description |
|---|---|
| **Data processing** | We used paired natural-language questions and SQL queries from the flight database. Before training, we replaced `<` with `<extra_id_1>` in SQL targets to prevent the comparison operator from becoming an unknown token. Natural-language inputs were left unchanged. Decoder inputs were constructed by prepending `<extra_id_0>` to the target sequence shifted one position to the right, enabling teacher forcing. Sequences were dynamically padded within each batch, with attention masks distinguishing actual tokens from padding. |
| **Tokenization** | We used `T5TokenizerFast` from `google-t5/t5-small` for both natural-language inputs and SQL outputs. The special token `<extra_id_0>` was used as the decoder start token. During decoding, we set `clean_up_tokenization_spaces=False` to preserve SQL formatting for exact-match evaluation, restored `<extra_id_1>` to `<`, and removed decoding-related special tokens to produce executable SQL queries. |
| **Architecture** | We initialized the pretrained `google-t5/t5-small` encoder-decoder model with its existing language-modeling head. We fine-tuned all model parameters without freezing layers or adding architectural components, allowing both the encoder and decoder to adapt to text-to-SQL generation. |
| **Hyperparameters** | We used AdamW with learning rate **3e-4**, weight decay **0**, and a cosine learning-rate scheduler with **one warmup epoch**. Training used batch size **16**, a maximum of **20 epochs**, and early stopping with patience **3**. The checkpoint with the highest development Record F1 was selected at epoch **14**, and training stopped after epoch **17**. For inference, we used beam search with **4 beams** and a maximum of **520 new tokens**. Increasing the learning rate from 1e-4 to 3e-4 improved development Record F1 from **58.63 to 74.42** under greedy decoding. The beam-search configuration further improved F1 to **82.45**, although this experiment also increased the generation cap from 350 to 520 tokens, preventing truncation of 47 long development queries. Therefore, the improvement cannot be attributed solely to beam search. |

## Q6. Results and error analysis (10 points)

**Table 4: Development and test results.**

| System | Query EM | F1 score |
|---|---|---|
| **Dev Results** | | |
| *T5 fine-tuned* | | |
| Full model (lr 3e-4, beam search with 4 beams, max 520 new tokens) | 59.23 | 82.45 |
| Baseline: lr 1e-4, greedy decoding, max 350 new tokens | 34.12 | 58.63 |
| Variant: lr 3e-4, greedy decoding, max 350 new tokens | 54.51 | 74.42 |
| Variant: lr 3e-4, beam search (4 beams), max 520 new tokens | 59.23 | 82.45 |
| **Test Results** | | |
| T5 fine-tuning | 75.46* | 79.14 |

Dev Query EM is exact match on the SQL string. \*The test leaderboard reports only Record EM (exact match on the returned database records), so the test Query EM entry is Record EM.

**Table 5: Qualitative error analysis on the dev set.**

| Error Type | Example of Error | Error Description | Statistics |
|---|---|---|---|
| SQL execution failure | #23: Flights arriving around 4 p.m. Gold: `arrival_time <= 1630`; predicted: `arrival_time< = 1630` | Invalid SQL caused by malformed operators, structural errors, invalid column references, or timeouts. | 43/466 |
| Incorrect temporal constraints | #144: Flights after 7 a.m. Gold: `departure_time > 700`; predicted: `departure_time > 1900` | Confuses AM/PM, time ranges, or dates, excluding valid results. | 19/466 |
| Incorrect route or direction | #239: Reverse a Pittsburgh-to-Denver flight. Gold: Denver → Pittsburgh; predicted: Pittsburgh → Denver | Fails to interpret route reversals, intermediate stops, or return journeys. | 16/466 |
| Incorrect selected columns | #400: Show flight times. Gold selects `departure_time`, `arrival_time`; prediction selects only `departure_time` | Omits requested fields or returns the wrong database entity. | 13/466 |
| Missing or incorrect airline filter | #70: Nonstop America West flights. Gold includes `airline_code = 'HP'`; prediction omits it | Loses the airline constraint while preserving other filters. | 7/466 |

Example numbers are 1-indexed line numbers in `data/dev.nl` / `data/dev.sql`. Predictions are from the full model (`results/t5_ft_lr3e4_beam4_dev.sql`).

**How the counts were calculated.** The 43 execution failures break down into 13 malformed comparison-operator errors, 17 other SQL structure errors, 9 invalid or ambiguous column-reference errors, and 4 timeouts.

For the remaining categories, I compared SQL conditions among queries that executed but returned incorrect records. I excluded five examples whose gold SQL uses a placeholder-like `state_code = 'ASD'` query.

These categories can overlap and are not intended to account for every SQL mismatch.

**Most actionable finding.** The model repeatedly generates `< =` instead of `<=`. Correcting this spacing in SQL postprocessing could eliminate several syntax errors, although evaluation would need to be rerun to measure the improvement.

## Q7. Performance evaluation (25 points)

<!-- Graded from t5_ft_experiment_test.sql / t5_ft_experiment_test.pkl on the leaderboard. No written answer required beyond matching Table 4's test row. -->
