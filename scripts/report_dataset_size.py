"""Report row / sentence counts for every split of the multilingual hotel_reviews corpus.

rows      = number of instances in the JSON list.
sentences = number of distinct `sentence_id` values.

Writes a tab-separated `dataset_size_report.csv` at the repo root.
"""

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO_ROOT, "dataset", "hotel_reviews")
OUT = os.path.join(REPO_ROOT, "dataset_size_report.csv")

LANGS = ["indo", "sun", "min", "jav", "mad", "eng"]
FOLDERS = ["mvp", "mvp_aos"]
# (column prefix, filename)
SPLITS = [
    ("train", "train.json"),
    ("val", "dev.json"),
    ("val_aug", "dev_aug.json"),
    ("test", "test.json"),
    ("test_aug", "test_aug.json"),
]

HEADER = [
    "language", "folder",
    "train_rows", "train_sentences",
    "val_rows", "val_sentences",
    "val_aug_rows", "val_aug_sentences",
    "test_rows", "test_sentences",
    "test_aug_rows", "test_aug_sentences",
]


def load_json(path):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeError(f"could not decode {path}")


def counts(path):
    if not os.path.exists(path):
        return "", ""
    data = load_json(path)
    rows = len(data)
    sentences = len({r["sentence_id"] for r in data})
    return rows, sentences


def main():
    lines = ["\t".join(HEADER)]
    for lang in LANGS:
        for folder in FOLDERS:
            folder_path = os.path.join(BASE, lang, folder)
            if not os.path.isdir(folder_path):
                continue
            fields = [lang, folder]
            for _, filename in SPLITS:
                rows, sentences = counts(os.path.join(folder_path, filename))
                fields += [rows, sentences]
            lines.append("\t".join(str(x) for x in fields))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
