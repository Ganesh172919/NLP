# SRL-Based Question Answering System with PropBank

A comprehensive Semantic Role Labeling (SRL)-based Question Answering system using unified PropBank annotations. This project implements a complete NLP pipeline for extracting answers from text using semantic role analysis.

## Project Overview

This system implements:
- **SRL Parsing**: Identifies predicates and their arguments following PropBank 3.1 specification
- **Question Analysis**: Classifies questions and maps them to expected semantic roles
- **Answer Extraction**: Extracts answers by mapping question types to PropBank argument positions
- **Frame Chaining**: Multi-hop reasoning across predicate-argument structures
- **Self-Correcting QA**: Iterative refinement with confidence thresholds

## Features

### Core Components

1. **Data Loader** (`data_loader.py`)
   - CoNLL format parsing (2004/2005 SRL format)
   - PropBank frame file (XML) parsing
   - Disfluency handling (preserves EDITED nodes per PropBank docs)
   - Support for unified frame semantics

2. **SRL Parser** (`srl_parser.py`)
   - Predicate identification
   - Argument extraction
   - PropBank sense assignment
   - Auxiliary verb handling (have.01, be.03, do.01)

3. **Question Analyzer** (`question_analyzer.py`)
   - Question type classification (WHO, WHAT, WHERE, WHEN, WHY, HOW)
   - Query predicate extraction
   - Expected role mapping

4. **Answer Extractor** (`answer_extractor.py`)
   - Role-based answer extraction
   - Predicate matching
   - Confidence scoring

### Innovation Layers

1. **Frame-Chaining QA** (`frame_chaining.py`)
   - Multi-hop reasoning across predicate graphs
   - Shared argument traversal
   - Chain path reconstruction

2. **Self-Correcting QA** (`self_correcting_qa.py`)
   - Iterative refinement workflow
   - Multiple correction strategies
   - Confidence-based stopping

3. **Financial QA Support**
   - Domain-specific predicate handling
   - SEC filing analysis capability

## Installation

```bash
# Clone the repository
git clone https://github.com/Ganesh172919/NLP.git
cd NLP

# Install dependencies (Python 3.8+)
pip install -r requirements.txt
```

## Usage

### Demo Mode
```bash
python -m srl_qa.main --demo
```

### Interactive Mode
```bash
python -m srl_qa.main --interactive
```

### Evaluation Mode
```bash
python -m srl_qa.main --evaluate test_data.json
```

### Python API

```python
from srl_qa import SRLQAPipeline, Config

# Initialize pipeline
config = Config()
pipeline = SRLQAPipeline(config)

# Answer a question
passage = "The company announced record profits yesterday."
question = "Who announced record profits?"

result = pipeline.answer(question, passage)
print(f"Answer: {result.answer}")  # Output: "The company"
print(f"Confidence: {result.confidence}")

# Use frame chaining for complex questions
result = pipeline.answer_with_chaining(question, passage)

# Use self-correction for better accuracy
result = pipeline.answer_with_correction(question, passage)
```

## PropBank Argument Types

### Core Arguments (ARG0-ARG5)
| Argument | Description |
|----------|-------------|
| ARG0 | Agent/Proto-Agent |
| ARG1 | Patient/Proto-Patient/Theme |
| ARG2 | Instrument/Benefactive/Attribute |
| ARG3 | Starting point/Benefactive |
| ARG4 | Ending point |
| ARG5 | Direction (rare) |

### Modifier Arguments (ARGM-*)
| Argument | Description |
|----------|-------------|
| ARGM-LOC | Location |
| ARGM-TMP | Temporal |
| ARGM-MNR | Manner |
| ARGM-CAU | Cause |
| ARGM-PRP | Purpose |
| ARGM-DIR | Direction |
| ARGM-NEG | Negation |
| ARGM-EXT | Extent |

## Question Type Mapping

| Question Type | Primary Roles |
|--------------|---------------|
| WHO | ARG0, ARG1, ARG2 |
| WHAT | ARG1, ARG2, ARG0 |
| WHERE | ARGM-LOC |
| WHEN | ARGM-TMP |
| WHY | ARGM-CAU, ARGM-PRP |
| HOW | ARGM-MNR |

## Project Structure

```
NLP/
├── srl_qa/
│   ├── __init__.py
│   ├── config.py              # Configuration and data classes
│   ├── data_loader.py         # PropBank data loading
│   ├── srl_parser.py          # SRL parsing
│   ├── question_analyzer.py   # Question analysis
│   ├── answer_extractor.py    # Answer extraction
│   ├── frame_chaining.py      # Frame-chaining QA
│   ├── self_correcting_qa.py  # Self-correcting QA
│   ├── pipeline.py            # Main pipeline
│   ├── main.py                # CLI entry point
│   ├── data/                  # Data handling modules
│   ├── models/                # Model definitions
│   ├── evaluation/            # Evaluation metrics
│   │   └── metrics.py
│   └── utils/                 # Utility functions
├── requirements.txt
└── readme.md
```

## Evaluation

The system includes comprehensive evaluation:

- **Exact Match (EM)**: Exact string match accuracy
- **F1 Score**: Token-level F1 score
- **Role Accuracy**: Accuracy by semantic role
- **Baseline Comparison**: Keyword-matching baseline

Run evaluation:
```bash
python -m srl_qa.main --evaluate
```

## References

1. Palmer, M., Gildea, D., & Kingsbury, P. (2005). The Proposition Bank: An Annotated Corpus of Semantic Roles.
2. Gildea, D., & Jurafsky, D. (2002). Automatic Labeling of Semantic Roles.
3. He, L., et al. (2015). Question-Answer Driven Semantic Role Labeling.
4. PropBank 3.1 Documentation: https://github.com/propbank/propbank-release

## License

This project is for academic purposes.

## Author

SRL-QA Research Team

