
# NLP (coding-first)

This repo is a small, runnable set of NLP learning scripts.

## Setup (Windows PowerShell)

```powershell
Set-Location c:\Users\RAVIPRAKASH\NLP

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -r requirements.txt
```

## Run

Each script is meant to be executed directly.

```powershell
$env:PYTHONPATH = "$PWD\src"

python .\scripts\01_clean_and_tokenize.py
python .\scripts\02_tfidf_sentiment.py
python .\scripts\03_topic_modeling_lda.py
python .\scripts\04_char_ngram_language_model.py
```

Optional (only if you want NLTK datasets installed locally):

```powershell
python .\scripts\00_setup_nltk.py
```

## Layout

- `src/nlp_learning/` reusable utilities
- `scripts/` runnable exercises (the “lessons”)
- `data/` tiny CSV datasets used by scripts

