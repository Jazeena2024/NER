# 🏷️ Named Entity Recognition (NER) Experiment Framework

This project implements a Python-based program to demonstrate and evaluate **Named Entity Recognition (NER)** using different NLP algorithms: **SpaCy, Stanza, and Flair**.

The system identifies entities such as **Person, Location, Organization, Date, etc.**, processes text from a corpus, and computes evaluation metrics including **precision, recall, F1-score, TP, FP, and FN**.

---

## 📌 Features

- Demonstrates basic NER on a single sentence
- Processes text from a corpus file
- Allows limiting the number of processed lines
- Supports three NER algorithms:
  - SpaCy
  - Stanza
  - Flair
- Calculates performance metrics:
  - Overall accuracy
  - Per-entity-category accuracy (PERSON, LOCATION, ORGANIZATION, etc.)
- Measures runtime for each algorithm
- Saves results to:
  - CSV file (spreadsheet-friendly)
  - PDF file (graphs and summary)

---

## 🛠️ Requirements

- Python **3.11.2**

Install required libraries:

```bash
pip install spacy stanza flair pandas matplotlib fpdf
```

## ⬇️ Download Required Language Models

Run the following commands after installing the libraries:

```bash
python -m spacy download en_core_web_sm
```

```bash
python -c "import stanza; stanza.download('en')"
```

These commands download the language models required for SpaCy and Stanza to perform Named Entity Recognition.

---

## 📂 Input Data

The input corpus file should be a plain text file, for example:
```bash
GMB_dataset.txt
```
Each line in the file contains a sentence used for NER processing.

### Example format:
```bash
Barack Obama was born in Hawaii.
Microsoft is based in Redmond.
The Eiffel Tower is located in Paris.
```
The program reads the file line by line and processes only the number of lines specified in the command-line argument.

## ▶️ How to Run

```bash
python ner_runexperiment.py GMB_dataset.txt 200 stanza results_spacy.csv results_spacy.pdf
```
- Processes 200 lines from GMB_dataset.txt
- Uses Stanza as the NER algorithm
- Outputs results.csv with detailed NER data
- Outputs results.pdf with summary and visualizations
  
## 📥 Command Line Arguments


| Argument | Description |
|----------|-------------|
| `GMB_dataset.txt` | Input corpus file |
| `200` | Number of lines to process |
| `stanza` | Algorithm to use (`spacy`, `stanza`, or `flair`) |
| `results_spacy.csv` | Output CSV file |
| `results_spacy.pdf` | Output PDF file |

### Example

```bash
python ner_runexperiment.py GMB_dataset.txt 200 spacy results_spacy.csv resuls_spacy.pdf
```
## ⚙️ Program Workflow

- Displays a demo sentence and prints detected entities by category  
- Reads text from a corpus file  
- Limits processing using a configurable line count  
- Runs the selected NER algorithm (SpaCy, Stanza, or Flair)  
- Calculates:
  - True Positives (TP)
  - False Positives (FP)
  - False Negatives (FN)
  - Precision
  - Recall
  - F1-score  
- Records execution time for the selected algorithm  
- Saves results to CSV and PDF formats

---
## 📊 Output

### CSV File

The CSV file contains:

- Entity type  
- True Positives (TP)  
- False Positives (FP)  
- False Negatives (FN)  
- Precision  
- Recall  
- F1-score  

### PDF File

The PDF file contains:

- Summary tables of results  
- Graphs of F1-score by entity category  
- Runtime information
## 🧪 Supported Algorithms

| Algorithm | Description |
|-----------|-------------|
| SpaCy     | Fast and lightweight NER engine |
| Stanza    | Neural pipeline from Stanford NLP |
| Flair     | Contextual string embedding NER |

---

## 🎯 Educational Purpose

This project demonstrates:

- How Named Entity Recognition works  
- How to evaluate NER systems  
- How to compare multiple NER algorithms  
- How to scale from single-sentence processing to large corpora  
- How to export results for spreadsheet analysis and visualization  

---
## 📁 Project Structure

- ner_runexperiment.py  
- GMB_dataset.txt  
- paperspast_corpus-English.txt  
- paperspast_corpus-Maori.txt  
- results_spacy.csv  
- results_spacy.pdf  
- README.md
Here’s a simple Markdown snippet for the Author section of your README.md:

## Author

**Jazeena Sam**  
Date: February 2026
