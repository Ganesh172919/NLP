"""
Configuration Module for SRL-QA System

This module contains all configuration settings for the SRL-Based Question
Answering system, including paths, model parameters, and evaluation settings.

PropBank-specific configurations follow the documentation from:
https://github.com/propbank/propbank-release
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class Config:
    """
    Configuration class for the SRL-QA system.
    
    Attributes:
        propbank_path: Path to PropBank release data
        treebank_path: Path to LDC Treebank data (Ontonotes 5.0, EWT)
        output_path: Path for output files
        model_name: Pre-trained model for SRL
        confidence_threshold: Minimum confidence for answer acceptance
        max_chain_depth: Maximum depth for frame chaining
        use_disfluency: Whether to preserve EDITED nodes (per PropBank docs)
    """
    
    # Data Paths
    propbank_path: str = "./data/propbank-release"
    treebank_path: str = "./data/treebank"
    output_path: str = "./output"
    frames_path: str = "./data/propbank-release/frames"
    
    # Model Configuration
    model_name: str = "bert-base-uncased"
    srl_model: str = "structured-prediction-srl-bert"
    
    # PropBank Argument Types (PropBank 3.1 specification)
    core_arguments: List[str] = field(default_factory=lambda: [
        "ARG0",  # Agent/Proto-Agent
        "ARG1",  # Patient/Proto-Patient/Theme
        "ARG2",  # Instrument/Benefactive/Attribute
        "ARG3",  # Starting point/Benefactive/Attribute
        "ARG4",  # Ending point
        "ARG5"   # Direction (rare)
    ])
    
    modifier_arguments: List[str] = field(default_factory=lambda: [
        "ARGM-LOC",  # Location
        "ARGM-TMP",  # Temporal
        "ARGM-MNR",  # Manner
        "ARGM-CAU",  # Cause
        "ARGM-PRP",  # Purpose
        "ARGM-DIR",  # Direction
        "ARGM-DIS",  # Discourse
        "ARGM-ADV",  # Adverbial
        "ARGM-MOD",  # Modal
        "ARGM-NEG",  # Negation
        "ARGM-EXT",  # Extent
        "ARGM-PRD",  # Secondary predication
        "ARGM-GOL",  # Goal
        "ARGM-COM",  # Comitative
        "ARGM-REC",  # Reciprocal
        "ARGM-LVB",  # Light verb
        "ARGM-CXN"   # Construction
    ])
    
    # Auxiliary Verbs (per PropBank documentation)
    auxiliary_verbs: Dict[str, str] = field(default_factory=lambda: {
        "have": "have.01",
        "be": "be.03",
        "do": "do.01"
    })
    
    # Question Type to Argument Mapping
    question_argument_map: Dict[str, List[str]] = field(default_factory=lambda: {
        "who": ["ARG0", "ARG1", "ARG2"],
        "what": ["ARG1", "ARG2", "ARG0"],
        "where": ["ARGM-LOC", "ARG2", "ARG4"],
        "when": ["ARGM-TMP"],
        "why": ["ARGM-CAU", "ARGM-PRP"],
        "how": ["ARGM-MNR", "ARGM-EXT"],
        "how much": ["ARGM-EXT", "ARG2"],
        "how many": ["ARGM-EXT", "ARG2"]
    })
    
    # Evaluation Settings
    test_split_path: str = "./data/propbank-release/docs/evaluation"
    min_test_questions: int = 100
    error_analysis_samples: int = 20
    
    # Frame Chaining Configuration
    max_chain_depth: int = 3
    confidence_threshold: float = 0.75
    self_correction_iterations: int = 3
    
    # Disfluency Handling (CRITICAL: per PropBank docs)
    preserve_edited_nodes: bool = True  # DO NOT remove EDITED nodes
    
    # Financial QA Configuration (for innovation layer)
    financial_wsj_path: str = "./data/wsj_financial"
    use_financial_domain: bool = False
    
    # Unified Frame Semantics
    enable_unified_frames: bool = True  # verbal/nominal/adjectival predicates
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create Config instance from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
    
    def to_dict(self) -> Dict:
        """Convert Config to dictionary."""
        return {
            'propbank_path': self.propbank_path,
            'treebank_path': self.treebank_path,
            'output_path': self.output_path,
            'model_name': self.model_name,
            'core_arguments': self.core_arguments,
            'modifier_arguments': self.modifier_arguments,
            'max_chain_depth': self.max_chain_depth,
            'confidence_threshold': self.confidence_threshold,
            'preserve_edited_nodes': self.preserve_edited_nodes
        }


@dataclass
class PropBankFrame:
    """
    Represents a PropBank frame definition.
    
    Per PropBank 3.1 specification, frames unify verbal, nominal, and
    adjectival predicates (e.g., "create" and "creation" share the same sense).
    
    Attributes:
        predicate: The predicate lemma (e.g., "create")
        sense: The sense number (e.g., "01")
        roleset_id: Full roleset ID (e.g., "create.01")
        roles: Dictionary mapping argument positions to role descriptions
        examples: List of example sentences
    """
    predicate: str
    sense: str
    roleset_id: str
    roles: Dict[str, str] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    
    # Unified frame types
    is_verbal: bool = True
    is_nominal: bool = False
    is_adjectival: bool = False
    
    def __str__(self) -> str:
        return f"{self.roleset_id}: {self.roles}"


@dataclass
class SemanticRole:
    """
    Represents a semantic role assignment.
    
    Attributes:
        role: The argument label (e.g., "ARG0", "ARGM-LOC")
        text: The text span assigned to this role
        start_idx: Start token index
        end_idx: End token index
        confidence: Model confidence score
    """
    role: str
    text: str
    start_idx: int
    end_idx: int
    confidence: float = 1.0
    
    def is_core_argument(self) -> bool:
        """Check if this is a core argument (ARG0-ARG5)."""
        return self.role.startswith("ARG") and not self.role.startswith("ARGM")
    
    def is_modifier(self) -> bool:
        """Check if this is a modifier argument (ARGM-*)."""
        return self.role.startswith("ARGM")


@dataclass
class PredicateArgumentStructure:
    """
    Represents a complete predicate-argument structure.
    
    This follows the PropBank annotation scheme where each predicate
    (verb, nominalization, or adjective) is associated with its arguments.
    
    Attributes:
        predicate: The predicate word
        predicate_sense: The PropBank sense (e.g., "create.01")
        arguments: List of semantic roles for this predicate
        sentence: The original sentence
    """
    predicate: str
    predicate_sense: str
    arguments: List[SemanticRole]
    sentence: str
    predicate_idx: int = 0
    
    def get_argument(self, role: str) -> Optional[SemanticRole]:
        """Get a specific argument by role label."""
        for arg in self.arguments:
            if arg.role == role:
                return arg
        return None
    
    def has_argument(self, role: str) -> bool:
        """Check if a specific argument exists."""
        return any(arg.role == role for arg in self.arguments)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'predicate': self.predicate,
            'sense': self.predicate_sense,
            'arguments': {arg.role: arg.text for arg in self.arguments},
            'sentence': self.sentence
        }
