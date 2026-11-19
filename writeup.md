---
title: "DS-GA 1011 Assignment 4: Fine-tuning Language Models"
author: "Maanas Lalwani"
geometry: margin=1in
fontsize: 11pt
---

## Links

- **GitHub repository (Part 1 and Part 2 code):** TODO
- **Google Drive folder (model checkpoint used for Q7 outputs):** TODO

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

**(1) Did performance on the transformed test data improve after data augmentation?**

TODO

**(2) How did data augmentation affect performance on the original test data?**

TODO

**Intuitive explanation of the results:**

TODO

**One limitation of this data augmentation approach for OOD test sets:**

TODO

# Part 2: Fine-tuning T5 for Text-to-SQL

## Q4. Data statistics and processing (5 points)

<!-- Use the T5 tokenizer (google-t5/t5-small) to compute these. -->

**Table 1: Data statistics before any pre-processing.**

| Statistics Name | Train | Dev |
|---|---|---|
| Number of examples | TODO | TODO |
| Mean sentence length | TODO | TODO |
| Mean SQL query length | TODO | TODO |
| Vocabulary size (natural language) | TODO | TODO |
| Vocabulary size (SQL) | TODO | TODO |

**Table 2: Data statistics after pre-processing.**

| Statistics Name | Train | Dev |
|---|---|---|
| **Model name** | TODO | |
| Mean sentence length | TODO | TODO |
| Mean SQL query length | TODO | TODO |
| Vocabulary size (natural language) | TODO | TODO |
| Vocabulary size (SQL) | TODO | TODO |

## Q5. T5 fine-tuning details (10 points)

**Table 3: Details of the best-performing T5 model configuration (fine-tuned).**

| Design choice | Description |
|---|---|
| Data processing | TODO |
| Tokenization | TODO |
| Architecture | TODO |
| Hyperparameters | TODO |

## Q6. Results and error analysis (10 points)

**Table 4: Development and test results.**

| System | Query EM | F1 score |
|---|---|---|
| **Dev Results** | | |
| *T5 fine-tuned* | | |
| Full model | TODO | TODO |
| Baseline: lr 1e-4, greedy decoding | 34.12 | 58.63 |
| Variant: lr 3e-4, greedy decoding | TODO | TODO |
| Variant: lr 3e-4, beam search (4 beams) | TODO | TODO |
| **Test Results** | | |
| T5 fine-tuning | TODO | TODO |

<!-- Baseline row is from the first run (dev SQL EM 0.3412, Record F1 0.5863). Update "Full model" to your final choice. -->

**Table 5: Qualitative error analysis on the dev set.**

| Error Type | Example of Error | Error Description | Statistics |
|---|---|---|---|
| TODO | TODO | TODO | TODO/466 |
| TODO | TODO | TODO | TODO/466 |
| TODO | TODO | TODO | TODO/466 |

## Q7. Performance evaluation (25 points)

<!-- Graded from t5_ft_experiment_test.sql / t5_ft_experiment_test.pkl on the leaderboard. No written answer required beyond matching Table 4's test row. -->
