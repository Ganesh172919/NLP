"""
SRL-Based Question Answering System with PropBank

This package implements a Semantic Role Labeling (SRL)-based Question Answering
system using unified PropBank annotations. It provides tools for:

1. Data Loading: Processing PropBank annotations in CoNLL format
2. SRL Parsing: Identifying predicates and their semantic arguments
3. Question Analysis: Understanding question types and extracting query predicates
4. Answer Extraction: Mapping questions to semantic roles for answer retrieval
5. Frame Chaining: Multi-hop reasoning across predicate-argument structures
6. Self-Correcting QA: Iterative refinement with confidence thresholds

Author: Academic Implementation
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "SRL-QA Research Team"

from .config import Config
from .data_loader import PropBankDataLoader
from .srl_parser import SRLParser
from .question_analyzer import QuestionAnalyzer
from .answer_extractor import AnswerExtractor
from .frame_chaining import FrameChainingQA
from .self_correcting_qa import SelfCorrectingQA
from .pipeline import SRLQAPipeline

__all__ = [
    'Config',
    'PropBankDataLoader',
    'SRLParser',
    'QuestionAnalyzer',
    'AnswerExtractor',
    'FrameChainingQA',
    'SelfCorrectingQA',
    'SRLQAPipeline'
]
