from pathlib import Path
from statistics import mean
from transformers import T5TokenizerFast
import sys

# Run from the part2 directory, or pass a data path
data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")

tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

def read_lines(split, extension):
    path = data_dir / f"{split}.{extension}"
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def calculate_stats(lines):
    lengths = []
    unique_tokens = set()

    for line in lines:
        token_ids = tokenizer.encode(line)
        lengths.append(len(token_ids))
        unique_tokens.update(token_ids)

    return {
        "mean_length": mean(lengths),
        "vocab_size": len(unique_tokens),
        "max_length": max(lengths),
    }

results = {}

for split in ["train", "dev"]:
    nl = read_lines(split, "nl")
    sql = read_lines(split, "sql")

    # Only SQL is preprocessed
    processed_sql = [
        query.replace("<", "<extra_id_1>")
        for query in sql
    ]

    results[split] = {
        "count": len(nl),
        "nl": calculate_stats(nl),
        "sql_before": calculate_stats(sql),
        "sql_after": calculate_stats(processed_sql),
        "queries_with_lt": sum("<" in query for query in sql),
    }

def print_table(title, rows):
    print(f"\n### {title}")
    print("| Statistic | Train | Dev |")
    print("|---|---:|---:|")

    for label, key in rows:
        values = []
        for split in ["train", "dev"]:
            value = key(results[split])
            if isinstance(value, float):
                value = f"{value:.2f}"
            values.append(str(value))

        print(f"| {label} | {values[0]} | {values[1]} |")

print_table("Table 1: Before preprocessing", [
    ("Number of examples", lambda r: r["count"]),
    ("Mean sentence length", lambda r: r["nl"]["mean_length"]),
    ("Mean SQL query length", lambda r: r["sql_before"]["mean_length"]),
    ("Vocabulary size (NL)", lambda r: r["nl"]["vocab_size"]),
    ("Vocabulary size (SQL)", lambda r: r["sql_before"]["vocab_size"]),
])

print_table("Table 2: After preprocessing", [
    ("Mean sentence length", lambda r: r["nl"]["mean_length"]),
    ("Mean SQL query length", lambda r: r["sql_after"]["mean_length"]),
    ("Vocabulary size (NL)", lambda r: r["nl"]["vocab_size"]),
    ("Vocabulary size (SQL)", lambda r: r["sql_after"]["vocab_size"]),
])

print("\nAdditional statistics:")
for split in ["train", "dev"]:
    r = results[split]
    print(f"{split}:")
    print(f"  SQL queries containing <: {r['queries_with_lt']}")
    print(f"  Max SQL length before: {r['sql_before']['max_length']}")
    print(f"  Max SQL length after:  {r['sql_after']['max_length']}")
