import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    qwerty_neighbors = {
        "a": ["q", "w", "s", "z"],
        "e": ["w", "s", "d", "r"],
        "i": ["u", "j", "k", "o"],
        "o": ["i", "k", "l", "p"],
        "u": ["y", "h", "j", "i"],
        "s": ["a", "w", "e", "d", "x", "z"],
        "t": ["r", "f", "g", "y"],
        "n": ["b", "h", "j", "m"],
        "r": ["e", "d", "f", "t"],
        "l": ["k", "o", "p"],
    }

    synonym_prob = 0.25  
    typo_prob = 0.30

    detokenizer = TreebankWordDetokenizer()
    tokens = word_tokenize(example["text"])
    transformed_tokens = []

    for token in tokens:
        if not token.isalpha() or len(token) <= 3:
            transformed_tokens.append(token)
            continue

        new_token = token

        if random.random() < synonym_prob:
            synonyms = set()
            for syn in wordnet.synsets(token.lower()):
                for lemma in syn.lemmas():
                    candidate = lemma.name().replace("_", " ")
                    if candidate.lower() != token.lower():
                        synonyms.add(candidate)

            if synonyms:
                chosen = random.choice(list(synonyms))
                if token[0].isupper():
                    chosen = chosen.capitalize()
                new_token = chosen

        if " " not in new_token and random.random() < typo_prob:
            chars = list(new_token)
            eligible_indices = [
                i for i, ch in enumerate(chars) if ch.lower() in qwerty_neighbors
            ]
            if eligible_indices:
                idx = random.choice(eligible_indices)
                orig_char = chars[idx]
                replacement = random.choice(qwerty_neighbors[orig_char.lower()])
                chars[idx] = (
                    replacement.upper() if orig_char.isupper() else replacement
                )
                new_token = "".join(chars)

        transformed_tokens.append(new_token)

    example["text"] = detokenizer.detokenize(transformed_tokens)

    return example
