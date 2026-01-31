"""
Semantic Role Labeling (SRL) Parser Module

This module provides SRL parsing functionality for the QA system,
supporting both neural-based models and rule-based fallback.

The parser identifies predicates and their arguments following PropBank
annotation guidelines, with support for:
- Verbal predicates
- Nominal predicates (unified frames)
- Adjectival predicates (unified frames)
- Auxiliary verbs (have.01, be.03, do.01)

References:
- He et al. (2017) Deep Semantic Role Labeling
- Gildea & Jurafsky (2002) Automatic Labeling of Semantic Roles
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .config import Config, SemanticRole, PredicateArgumentStructure

logger = logging.getLogger(__name__)


@dataclass
class Token:
    """Represents a tokenized word with its properties."""
    text: str
    lemma: str
    pos: str
    index: int
    is_predicate: bool = False
    predicate_sense: Optional[str] = None


class SRLParser:
    """
    Semantic Role Labeling parser for PropBank-style annotations.
    
    This parser supports:
    1. Neural SRL using pre-trained models (when available)
    2. Rule-based SRL as fallback
    3. PropBank frame lookup for role descriptions
    4. Unified frame handling for verbal/nominal/adjectival predicates
    
    Usage:
        parser = SRLParser(config)
        structures = parser.parse("The company announced profits.")
    """
    
    def __init__(self, config: Config):
        """
        Initialize the SRL parser.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.frames: Dict[str, Dict] = {}
        self._model = None
        self._tokenizer = None
        
        # Common predicate patterns for rule-based parsing
        self.verb_patterns = [
            r'\b(announce[sd]?|announc(?:ing|ed))\b',
            r'\b(declin(?:e[sd]?|ing))\b',
            r'\b(increas(?:e[sd]?|ing))\b',
            r'\b(stat(?:e[sd]?|ing))\b',
            r'\b(creat(?:e[sd]?|ing))\b',
            r'\b(express(?:ed|ing|es)?)\b',
            r'\b(approv(?:e[sd]?|ing))\b',
            r'\b(report(?:ed|ing|s)?)\b',
            r'\b(caus(?:e[sd]?|ing))\b',
            r'\b(affect(?:ed|ing|s)?)\b',
        ]
        
        # Argument patterns
        self.arg0_patterns = [
            r'^(The\s+\w+)',
            r'^(\w+\s+(?:company|corporation|firm|CEO|board|investors))',
            r'^(He|She|They|We|I)',
        ]
        
        self.temporal_patterns = [
            r'(yesterday|today|tomorrow)',
            r'(last\s+(?:week|month|year|quarter))',
            r'(this\s+(?:week|month|year|quarter))',
            r'(in\s+\d{4})',
            r'(on\s+\w+day)',
            r'(in\s+Q[1-4])',
        ]
        
        self.causal_patterns = [
            r'(because\s+of\s+.+)',
            r'(due\s+to\s+.+)',
            r'(owing\s+to\s+.+)',
            r'(as\s+a\s+result\s+of\s+.+)',
        ]
        
        self.manner_patterns = [
            r'(\w+ly)',  # Adverbs
            r'(quickly|slowly|sharply|gradually|unanimously)',
        ]
        
        self.location_patterns = [
            r'(in\s+(?:the\s+)?\w+(?:\s+\w+)?(?:\'s)?\s+(?:meeting|office|building))',
            r'(at\s+(?:the\s+)?\w+)',
            r'(in\s+(?:New\s+York|London|Tokyo|Chicago))',
        ]
        
        # Auxiliary verb mappings per PropBank docs
        self.auxiliary_mappings = {
            'have': 'have.01',
            'has': 'have.01',
            'had': 'have.01',
            'having': 'have.01',
            'be': 'be.03',
            'is': 'be.03',
            'are': 'be.03',
            'was': 'be.03',
            'were': 'be.03',
            'being': 'be.03',
            'been': 'be.03',
            'do': 'do.01',
            'does': 'do.01',
            'did': 'do.01',
            'doing': 'do.01',
        }
        
        # Predicate to sense mapping
        self.predicate_senses = {
            'announce': 'announce.01',
            'announced': 'announce.01',
            'announces': 'announce.01',
            'announcing': 'announce.01',
            'decline': 'decline.01',
            'declined': 'decline.01',
            'declines': 'decline.01',
            'declining': 'decline.01',
            'increase': 'increase.01',
            'increased': 'increase.01',
            'increases': 'increase.01',
            'increasing': 'increase.01',
            'state': 'state.01',
            'stated': 'state.01',
            'states': 'state.01',
            'stating': 'state.01',
            'create': 'create.01',
            'created': 'create.01',
            'creates': 'create.01',
            'creating': 'create.01',
            'creation': 'create.01',  # Nominal form (unified)
            'express': 'express.01',
            'expressed': 'express.01',
            'expresses': 'express.01',
            'expressing': 'express.01',
            'approve': 'approve.01',
            'approved': 'approve.01',
            'approves': 'approve.01',
            'approving': 'approve.01',
            'approval': 'approve.01',  # Nominal form (unified)
            'report': 'report.01',
            'reported': 'report.01',
            'reports': 'report.01',
            'reporting': 'report.01',
            'cause': 'cause.01',
            'caused': 'cause.01',
            'causes': 'cause.01',
            'causing': 'cause.01',
            'delay': 'delay.01',
            'delayed': 'delay.01',
            'delays': 'delay.01',
            'delaying': 'delay.01',
        }
    
    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenize input text.
        
        Args:
            text: Input sentence
            
        Returns:
            List of Token objects
        """
        words = text.split()
        tokens = []
        
        for idx, word in enumerate(words):
            # Simple lemmatization
            lemma = word.lower().rstrip('.,!?;:')
            
            # Simple POS tagging
            pos = self._simple_pos_tag(word)
            
            # Check if predicate
            is_pred = lemma in self.predicate_senses or lemma in self.auxiliary_mappings
            pred_sense = None
            if is_pred:
                pred_sense = self.predicate_senses.get(
                    lemma, 
                    self.auxiliary_mappings.get(lemma)
                )
            
            tokens.append(Token(
                text=word,
                lemma=lemma,
                pos=pos,
                index=idx,
                is_predicate=is_pred,
                predicate_sense=pred_sense
            ))
            
        return tokens
    
    def _simple_pos_tag(self, word: str) -> str:
        """Simple POS tagger based on patterns."""
        word_lower = word.lower()
        
        # Verbs
        if word_lower.endswith(('ed', 'ing', 'es', 's')):
            if word_lower in self.predicate_senses:
                return 'VB'
        
        # Nouns
        if word_lower.endswith(('tion', 'ment', 'ness', 'ity', 'er', 'or')):
            return 'NN'
        
        # Adjectives
        if word_lower.endswith(('ly',)):
            return 'RB'  # Adverb
        
        if word_lower.endswith(('able', 'ible', 'ful', 'less')):
            return 'JJ'
        
        # Determiners
        if word_lower in ('the', 'a', 'an', 'this', 'that', 'these', 'those'):
            return 'DT'
        
        # Prepositions
        if word_lower in ('in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'about', 'of'):
            return 'IN'
        
        # Default
        return 'NN'
    
    def parse(self, text: str) -> List[PredicateArgumentStructure]:
        """
        Parse a sentence and extract predicate-argument structures.
        
        This method uses rule-based parsing following PropBank guidelines.
        
        Args:
            text: Input sentence
            
        Returns:
            List of PredicateArgumentStructure objects
        """
        tokens = self.tokenize(text)
        structures = []
        
        # Find predicates
        predicate_indices = [t.index for t in tokens if t.is_predicate]
        
        if not predicate_indices:
            # Try pattern-based predicate detection
            predicate_indices = self._find_predicates_by_pattern(text, tokens)
        
        for pred_idx in predicate_indices:
            if pred_idx >= len(tokens):
                continue
                
            pred_token = tokens[pred_idx]
            
            # Get predicate sense
            pred_sense = pred_token.predicate_sense or self._get_predicate_sense(
                pred_token.lemma
            )
            
            # Extract arguments
            arguments = self._extract_arguments(text, tokens, pred_idx)
            
            structures.append(PredicateArgumentStructure(
                predicate=pred_token.text,
                predicate_sense=pred_sense,
                arguments=arguments,
                sentence=text,
                predicate_idx=pred_idx
            ))
        
        return structures
    
    def _find_predicates_by_pattern(
        self, 
        text: str, 
        tokens: List[Token]
    ) -> List[int]:
        """Find predicates using regex patterns."""
        predicate_indices = []
        text_lower = text.lower()
        
        for pattern in self.verb_patterns:
            match = re.search(pattern, text_lower)
            if match:
                matched_word = match.group(1)
                for token in tokens:
                    if token.lemma == matched_word.lower() or token.text.lower() == matched_word:
                        predicate_indices.append(token.index)
                        break
        
        return list(set(predicate_indices))
    
    def _get_predicate_sense(self, lemma: str) -> str:
        """Get PropBank sense for a predicate."""
        return self.predicate_senses.get(lemma, f"{lemma}.01")
    
    def _extract_arguments(
        self, 
        text: str, 
        tokens: List[Token], 
        pred_idx: int
    ) -> List[SemanticRole]:
        """
        Extract arguments for a predicate using pattern matching.
        
        Args:
            text: Original sentence
            tokens: Tokenized sentence
            pred_idx: Index of the predicate
            
        Returns:
            List of SemanticRole objects
        """
        arguments = []
        text_lower = text.lower()
        
        # ARG0 - Agent (typically before the predicate)
        arg0_text = self._find_arg0(text, tokens, pred_idx)
        if arg0_text:
            arguments.append(SemanticRole(
                role='ARG0',
                text=arg0_text,
                start_idx=0,
                end_idx=pred_idx - 1
            ))
        
        # ARG1 - Patient/Theme (typically after the predicate)
        arg1_text = self._find_arg1(text, tokens, pred_idx)
        if arg1_text:
            arguments.append(SemanticRole(
                role='ARG1',
                text=arg1_text,
                start_idx=pred_idx + 1,
                end_idx=len(tokens) - 1
            ))
        
        # ARGM-TMP - Temporal
        for pattern in self.temporal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                arguments.append(SemanticRole(
                    role='ARGM-TMP',
                    text=match.group(1),
                    start_idx=-1,
                    end_idx=-1
                ))
                break
        
        # ARGM-CAU - Cause
        for pattern in self.causal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                cause_text = match.group(1)
                # Clean up the cause text
                cause_text = re.sub(r'^(because of|due to|owing to|as a result of)\s+', '', cause_text, flags=re.IGNORECASE)
                arguments.append(SemanticRole(
                    role='ARGM-CAU',
                    text=cause_text,
                    start_idx=-1,
                    end_idx=-1
                ))
                break
        
        # ARGM-MNR - Manner
        for pattern in self.manner_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                manner = match.group(1)
                if manner.lower() not in ('the', 'a', 'an'):  # Filter false positives
                    arguments.append(SemanticRole(
                        role='ARGM-MNR',
                        text=manner,
                        start_idx=-1,
                        end_idx=-1
                    ))
                    break
        
        # ARGM-LOC - Location
        for pattern in self.location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                arguments.append(SemanticRole(
                    role='ARGM-LOC',
                    text=match.group(1),
                    start_idx=-1,
                    end_idx=-1
                ))
                break
        
        return arguments
    
    def _find_arg0(self, text: str, tokens: List[Token], pred_idx: int) -> Optional[str]:
        """Find the ARG0 (agent) of a predicate."""
        if pred_idx == 0:
            return None
        
        # Get tokens before predicate
        pre_tokens = tokens[:pred_idx]
        
        # Look for noun phrase at the beginning
        for pattern in self.arg0_patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Fallback: take all tokens before predicate
        if pre_tokens:
            arg0_words = [t.text for t in pre_tokens]
            return ' '.join(arg0_words).strip()
        
        return None
    
    def _find_arg1(self, text: str, tokens: List[Token], pred_idx: int) -> Optional[str]:
        """Find the ARG1 (patient/theme) of a predicate."""
        if pred_idx >= len(tokens) - 1:
            return None
        
        # Get tokens after predicate
        post_tokens = tokens[pred_idx + 1:]
        
        # Filter out modifiers
        arg1_words = []
        for token in post_tokens:
            # Stop at temporal/causal markers
            if token.text.lower() in ('yesterday', 'today', 'because', 'due', 'in', 'on', 'at'):
                # Check if this is part of the object or a modifier
                if token.text.lower() in ('in', 'on', 'at'):
                    # Could be part of object or location - be conservative
                    break
                if token.text.lower() in ('yesterday', 'today', 'because', 'due'):
                    break
            arg1_words.append(token.text)
        
        if arg1_words:
            return ' '.join(arg1_words).strip().rstrip('.,!?;:')
        
        return None
    
    def parse_with_frames(
        self, 
        text: str, 
        frames: Dict[str, 'PropBankFrame']
    ) -> List[PredicateArgumentStructure]:
        """
        Parse a sentence using PropBank frame definitions.
        
        Args:
            text: Input sentence
            frames: Dictionary of PropBank frames
            
        Returns:
            List of PredicateArgumentStructure objects
        """
        structures = self.parse(text)
        
        # Enhance with frame information
        for structure in structures:
            if structure.predicate_sense in frames:
                frame = frames[structure.predicate_sense]
                # Add role descriptions from frame
                for arg in structure.arguments:
                    if arg.role in frame.roles:
                        arg.confidence = 1.0  # Boost confidence for frame-matched roles
        
        return structures
    
    def get_unified_frames(self, predicate: str) -> List[str]:
        """
        Get unified frames for a predicate (verbal/nominal/adjectival).
        
        Per PropBank 3.1, predicates like "create" and "creation" share
        the same roleset structure.
        
        Args:
            predicate: Predicate word
            
        Returns:
            List of related roleset IDs
        """
        base_predicate = predicate.lower().rstrip('ed').rstrip('ing').rstrip('s')
        
        related = []
        for pred, sense in self.predicate_senses.items():
            if pred.startswith(base_predicate) or base_predicate.startswith(pred.rstrip('e')):
                related.append(sense)
        
        return list(set(related))
    
    def handle_auxiliary_verbs(self, tokens: List[Token]) -> List[Token]:
        """
        Handle auxiliary verbs per PropBank documentation.
        
        Per PropBank docs, auxiliary verbs have specific rolesets:
        - have.01 for 'have' as auxiliary
        - be.03 for 'be' as auxiliary
        - do.01 for 'do' as auxiliary
        
        Args:
            tokens: List of tokens
            
        Returns:
            Updated token list with auxiliary annotations
        """
        for i, token in enumerate(tokens):
            if token.lemma in self.auxiliary_mappings:
                # Check if it's being used as auxiliary
                if i + 1 < len(tokens):
                    next_token = tokens[i + 1]
                    if next_token.pos == 'VB' or next_token.text.endswith(('ing', 'ed')):
                        # This is an auxiliary
                        token.predicate_sense = self.auxiliary_mappings[token.lemma]
                        token.is_predicate = True
        
        return tokens
