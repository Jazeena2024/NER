import spacy
import stanza
from flair.models import SequenceTagger
from flair.data import Sentence
import pandas as pd
from collections import defaultdict
import sys
import os
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict

# -----------------------
# CHECK COMMAND LINE
# -----------------------
if len(sys.argv) != 6:
    print("Usage:")
    print("python ner_runexperiment.py <corpus_file> <line_limit> <algorithm> <csv_file> <pdf_file>")
    sys.exit(1)

CORPUS_FILE = sys.argv[1]
LINE_LIMIT = int(sys.argv[2])
ALGORITHM = sys.argv[3].lower()
CSV_FILE = sys.argv[4]
PDF_FILE = sys.argv[5]

if not os.path.exists(CORPUS_FILE):
    print(f"Error: Corpus file not found: {CORPUS_FILE}")
    sys.exit(1)

# -----------------------
# AUTO-RENAME OUTPUT FILES (ADD ALGORITHM NAME)
# -----------------------
base_csv, csv_ext = os.path.splitext(CSV_FILE)
base_pdf, pdf_ext = os.path.splitext(PDF_FILE)

CSV_FILE = f"{base_csv}_{ALGORITHM}{csv_ext}"
PDF_FILE = f"{base_pdf}_{ALGORITHM}{pdf_ext}"

print("Corpus file:", CORPUS_FILE)
print("Line limit:", LINE_LIMIT)
print("Algorithm:", ALGORITHM)
print("CSV output:", CSV_FILE)
print("PDF output:", PDF_FILE)

# -----------------------
# START TIMING
# -----------------------
start_time = time.perf_counter()

# -----------------------
# LOAD MODEL
# -----------------------
print("Loading model...")

if ALGORITHM == "spacy":
    spacy_nlp = spacy.load("en_core_web_sm")
elif ALGORITHM == "stanza":
    stanza.download("en")
    stanza_nlp = stanza.Pipeline("en", processors="tokenize,ner", tokenize_no_ssplit=True)
elif ALGORITHM == "flair":
    flair_tagger = SequenceTagger.load("ner")
else:
    print("Invalid algorithm. Choose: spacy, stanza, or flair")
    sys.exit(1)

# -----------------------
# TAG MAPPING
# -----------------------
TAG_MAP = {
    "geo": "GPE",
    "gpe": "GPE",
    "per": "PERSON",
    "org": "ORG",
    "tim": "DATE",
    "loc": "LOC"
}

# -----------------------
# READ DATASET (SAFE)
# -----------------------
columns = ["sentence_id", "word", "pos", "tag"]

try:
    df = pd.read_csv(
        CORPUS_FILE,
        sep="\t",
        names=columns,
        encoding="latin1",
        engine="python",       # Use Python engine to avoid C parser errors
        on_bad_lines="skip"    # Skip broken/malformed lines
    )
except Exception as e:
    print("Error reading dataset:", e)
    sys.exit(1)

df = df.head(LINE_LIMIT)

# -----------------------
# RECONSTRUCT SENTENCES
# -----------------------
sentences = []
current_sentence = []
current_sid = df.iloc[0]["sentence_id"] if not df.empty else 0

for _, row in df.iterrows():
    if row["sentence_id"] != current_sid:
        sentences.append(" ".join(current_sentence))
        current_sentence = []
        current_sid = row["sentence_id"]
    current_sentence.append(str(row["word"]))

if current_sentence:
    sentences.append(" ".join(current_sentence))

print("Total sentences:", len(sentences))

# -----------------------
# GOLD ENTITIES
# -----------------------
gold_entities = defaultdict(set)
current_entity = []
current_label = None

for word, tag in zip(df["word"], df["tag"]):
    if isinstance(tag, str) and tag.startswith("B-"):
        if current_entity:
            gold_entities[current_label].add(" ".join(current_entity))
        label = tag.split("-")[1]
        current_label = TAG_MAP.get(label.lower(), label.upper())
        current_entity = [word]

    elif isinstance(tag, str) and tag.startswith("I-") and current_label:
        current_entity.append(word)

    else:
        if current_entity:
            gold_entities[current_label].add(" ".join(current_entity))
        current_entity = []
        current_label = None

if current_entity:
    gold_entities[current_label].add(" ".join(current_entity))

# Define allowed entity types
ALLOWED_ENTITIES = {"PERSON", "ORG", "LOC", "GPE", "DATE"}


# -----------------------
# NER FUNCTIONS
# -----------------------
def run_spacy(sentences):
    pred_entities = defaultdict(set)
    for sent in sentences:
        doc = spacy_nlp(sent)
        for ent in doc.ents:
            if ent.label_ in ALLOWED_ENTITIES:
                pred_entities[ent.label_].add(ent.text)
    return pred_entities

def run_stanza(sentences):
    pred_entities = defaultdict(set)
    for sent in sentences:
        doc = stanza_nlp(sent)
        for ent in doc.ents:
            if ent.type in ALLOWED_ENTITIES:
                pred_entities[ent.type].add(ent.text)
    return pred_entities

def run_flair(sentences):
    pred_entities = defaultdict(set)
    for sent in sentences:
        flair_sent = Sentence(sent)
        flair_tagger.predict(flair_sent)
        for ent in flair_sent.get_spans("ner"):
            if ent.tag in ALLOWED_ENTITIES:
                pred_entities[ent.tag].add(ent.text)
    return pred_entities
# -----------------------
# RUN SELECTED ALGORITHM
# -----------------------
print(f"\nRunning {ALGORITHM.upper()}...")

ner_start = time.perf_counter()

if ALGORITHM == "spacy":
    pred_entities = run_spacy(sentences)
elif ALGORITHM == "stanza":
    pred_entities = run_stanza(sentences)
elif ALGORITHM == "flair":
    pred_entities = run_flair(sentences)

ner_time = time.perf_counter() - ner_start

# -----------------------
# METRICS
# -----------------------
def compute_metrics(gold, pred):
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return tp, fp, fn, precision, recall, f1

rows = []

all_gold = set().union(*gold_entities.values()) if gold_entities else set()
all_pred = set().union(*pred_entities.values()) if pred_entities else set()

tp, fp, fn, p, r, f1 = compute_metrics(all_gold, all_pred)

labels = sorted(set(gold_entities) | set(pred_entities))

for label in labels:
    g = gold_entities.get(label, set())
    pr = pred_entities.get(label, set())
    tpi, fpi, fni, pi, ri, f1i = compute_metrics(g, pr)
    rows.append([label, tpi, fpi, fni, round(pi,4), round(ri,4), round(f1i,4)])

rows.append(["OVERALL", tp, fp, fn, round(p,4), round(r,4), round(f1,4)])

results_df = pd.DataFrame(
    rows,
    columns=["Entity", "TP", "FP", "FN", "Precision", "Recall", "F1"]
)

# -----------------------
# SAVE CSV
# -----------------------
results_df.to_csv(CSV_FILE, index=False)
print("CSV saved:", CSV_FILE)

# -----------------------
# SAVE PDF
# -----------------------
with PdfPages(PDF_FILE) as pdf:
    fig, ax = plt.subplots(figsize=(8, len(rows)*0.4 + 1))
    ax.axis("off")
    ax.table(
        cellText=results_df.values,
        colLabels=results_df.columns,
        loc="center"
    )
    plt.title(f"{ALGORITHM.upper()} NER Results")
    pdf.savefig(fig)
    plt.close()

    plot_df = results_df[results_df["Entity"] != "OVERALL"]
    plot_df.set_index("Entity")[["Precision", "Recall", "F1"]].plot(kind="bar", figsize=(8,4))
    plt.title(f"{ALGORITHM.upper()} NER Performance by Entity")
    plt.ylabel("Score")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print("PDF saved:", PDF_FILE)
print("NER time:", round(ner_time,4), "seconds")

# -----------------------
# TOTAL TIME
# -----------------------
total_time = time.perf_counter() - start_time
print("\nTotal runtime:", round(total_time,4), "seconds")
