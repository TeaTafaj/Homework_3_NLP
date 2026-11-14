import json
import random
from pathlib import Path

import nltk
from gensim.utils import simple_preprocess

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

TEXT_DIR = Path("Train_Test_Documents")  # folder with the 571 .txt files
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TARGET_WORDS = 120  # approximate chunk size
MIN_TOKENS = 5  # drop chunks shorter than this
MAX_TOKENS = 200  # drop chunks longer than this

# Keyword tiers (from tutorial step 6)
TIER_A = {"merchant", "merchants", "marchant", "marchants"}
TIER_B = {
    "factor",
    "chapman",
    "adventurer",
    "adventurers",
    "venturer",
    "venturers",
    "staple",
    "staplers",
    "trade",
    "purser",
}
TIER_C = {
    "clothier",
    "clothyer",
    "tailor",
    "tayler",
    "haberdasher",
    "goldsmith",
    "vintner",
    "brewer",
    "banker",
    "grazier",
    "jeweller",
}


def chunk_text(text, target=TARGET_WORDS):
    """Group sentences into chunks of roughly `target` words."""
    chunks, current, count = [], [], 0
    for sent in nltk.sent_tokenize(text):
        n = len(sent.split())
        if n == 0:
            continue
        if current and count + n > target:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def label_chunk(tokens):
    """1 = CORE (Tier A/B word present), 2 = MAYBE (only Tier C), 0 = NEG."""
    words = set(tokens)
    if words & (TIER_A | TIER_B):
        return 1
    if words & TIER_C:
        return 2
    return 0


# ---------- Label all chunks (tutorial step 7) ----------
files = sorted(TEXT_DIR.glob("*.txt"))
print("Text files found:", len(files))

labeled = []
for i, path in enumerate(files, start=1):
    text = path.read_text(encoding="utf-8", errors="ignore")
    for chunk in chunk_text(text):
        tokens = simple_preprocess(chunk, deacc=True)
        if MIN_TOKENS <= len(tokens) <= MAX_TOKENS:
            labeled.append((chunk, label_chunk(tokens)))
    if i % 50 == 0 or i == len(files):
        print(f"  processed {i}/{len(files)} files")

with open(DATA_DIR / "merchant_labeled_chunks.json", "w", encoding="utf-8") as f:
    json.dump(labeled, f, ensure_ascii=False)

# ---------- Balance and split (tutorial step 9) ----------
random.seed(42)

core = [(t, 1) for (t, y) in labeled if y == 1]
neg = [(t, 0) for (t, y) in labeled if y == 0]
maybe = [t for (t, y) in labeled if y == 2]

print("\nLabel counts:")
print("  CORE :", len(core))
print("  NEG  :", len(neg))
print("  MAYBE:", len(maybe))

neg_sample = random.sample(neg, len(core))  # same number of NEG as CORE
dataset = core + neg_sample
random.shuffle(dataset)

split = int(0.8 * len(dataset))
train_data = dataset[:split]
test_data = dataset[split:]

print("\nTrain size:", len(train_data))
print("Test size :", len(test_data))

with open(DATA_DIR / "train_core_vs_neg.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False)
with open(DATA_DIR / "test_core_vs_neg.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False)

print("\nSaved data/train_core_vs_neg.json and data/test_core_vs_neg.json")
