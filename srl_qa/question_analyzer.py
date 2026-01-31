"""
Question Analyzer Module

This module analyzes questions to extract query predicates and expected
answer types, enabling mapping to PropBank semantic roles.

The analyzer supports various question types:
- WH-questions (who, what, where, when, why, how)
- Yes/No questions
- Complex questions requiring frame chaining

References:
- He et al. (2015) Question-Answer Driven Semantic Role Labeling
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .config import Config

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Enumeration of supported question types."""
    WHO = "who"
    WHAT = "what"
    WHERE = "where"
    WHEN = "when"
    WHY = "why"
    HOW = "how"
    HOW_MUCH = "how_much"
    HOW_MANY = "how_many"
    WHICH = "which"
    YES_NO = "yes_no"
    UNKNOWN = "unknown"


@dataclass
class AnalyzedQuestion:
    """
    Represents an analyzed question with extracted components.
    
    Attributes:
        original: The original question text
        question_type: Classified question type
        expected_roles: PropBank roles that likely contain the answer
        query_predicate: The main predicate from the question
        query_arguments: Arguments extracted from the question
        focus_entities: Named entities or key terms in focus
        confidence: Analysis confidence score
    """
    original: str
    question_type: QuestionType
    expected_roles: List[str]
    query_predicate: Optional[str] = None
    query_arguments: Dict[str, str] = field(default_factory=dict)
    focus_entities: List[str] = field(default_factory=list)
    confidence: float = 1.0
    requires_chaining: bool = False


class QuestionAnalyzer:
    """
    Analyzes questions to enable SRL-based answer extraction.
    
    The analyzer performs:
    1. Question type classification
    2. Query predicate extraction
    3. Expected answer role mapping
    4. Multi-hop question detection
    
    Usage:
        analyzer = QuestionAnalyzer(config)
        analysis = analyzer.analyze("Who announced the profits?")
        print(analysis.expected_roles)  # ['ARG0']
    """
    
    def __init__(self, config: Config):
        """
        Initialize the question analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Question word patterns
        self.question_patterns = {
            QuestionType.WHO: [
                r'^who\s+',
                r'\bwho\b',
            ],
            QuestionType.WHAT: [
                r'^what\s+',
                r'\bwhat\b',
            ],
            QuestionType.WHERE: [
                r'^where\s+',
                r'\bwhere\b',
            ],
            QuestionType.WHEN: [
                r'^when\s+',
                r'\bwhen\b',
            ],
            QuestionType.WHY: [
                r'^why\s+',
                r'\bwhy\b',
            ],
            QuestionType.HOW_MUCH: [
                r'^how\s+much\b',
                r'\bhow\s+much\b',
            ],
            QuestionType.HOW_MANY: [
                r'^how\s+many\b',
                r'\bhow\s+many\b',
            ],
            QuestionType.HOW: [
                r'^how\s+(?!much|many)',
                r'\bhow\b(?!\s+(?:much|many))',
            ],
            QuestionType.WHICH: [
                r'^which\s+',
                r'\bwhich\b',
            ],
        }
        
        # Question type to expected semantic role mapping
        self.role_mapping = {
            QuestionType.WHO: ['ARG0', 'ARG1', 'ARG2'],
            QuestionType.WHAT: ['ARG1', 'ARG2', 'ARG0'],
            QuestionType.WHERE: ['ARGM-LOC', 'ARG2', 'ARG4'],
            QuestionType.WHEN: ['ARGM-TMP'],
            QuestionType.WHY: ['ARGM-CAU', 'ARGM-PRP'],
            QuestionType.HOW: ['ARGM-MNR'],
            QuestionType.HOW_MUCH: ['ARGM-EXT', 'ARG2'],
            QuestionType.HOW_MANY: ['ARGM-EXT', 'ARG2'],
            QuestionType.WHICH: ['ARG1', 'ARG0'],
            QuestionType.YES_NO: ['ARG1', 'ARG0'],
            QuestionType.UNKNOWN: ['ARG1', 'ARG0', 'ARG2'],
        }
        
        # Predicate indicators in questions
        self.predicate_patterns = [
            r'\b(cause[sd]?|causing)\b',
            r'\b(announce[sd]?|announcing)\b',
            r'\b(report(?:ed|ing|s)?)\b',
            r'\b(declin(?:e[sd]?|ing))\b',
            r'\b(increas(?:e[sd]?|ing))\b',
            r'\b(affect(?:ed|ing|s)?)\b',
            r'\b(creat(?:e[sd]?|ing))\b',
            r'\b(stat(?:e[sd]?|ing))\b',
            r'\b(happen(?:ed|ing|s)?)\b',
            r'\b(occur(?:red|ring|s)?)\b',
            r'\b(result(?:ed|ing|s)?)\b',
            r'\b(lead(?:ing)?|led)\b',
            r'\b(express(?:ed|ing|es)?)\b',
            r'\b(approv(?:e[sd]?|ing))\b',
        ]
        
        # Chaining indicators (questions requiring multi-hop reasoning)
        self.chaining_patterns = [
            r'\b(because\s+of|due\s+to|result(?:ed)?\s+in)\b',
            r'\b(led\s+to|caused\s+by)\b',
            r'\b(after|before|when)\s+\w+\s+(happen|occur)',
            r'\b(and\s+(?:then|also|subsequently))\b',
        ]
        
        # Focus entity patterns
        self.entity_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Capitalized names
            r'\b(the\s+\w+)\b',  # The + noun
            r'\"([^\"]+)\"',  # Quoted text
            r'\$[\d,]+(?:\.\d{2})?',  # Money amounts
            r'\d+%',  # Percentages
        ]
    
    def analyze(self, question: str) -> AnalyzedQuestion:
        """
        Analyze a question and extract structured information.
        
        Args:
            question: The input question
            
        Returns:
            AnalyzedQuestion object with extracted components
        """
        question_lower = question.lower()
        
        # Classify question type
        q_type = self._classify_question(question_lower)
        
        # Get expected roles
        expected_roles = self.role_mapping.get(q_type, ['ARG1'])
        
        # Extract query predicate
        query_pred = self._extract_predicate(question)
        
        # Extract focus entities
        focus_entities = self._extract_entities(question)
        
        # Check if chaining is required
        requires_chaining = self._requires_chaining(question_lower)
        
        # Extract query arguments
        query_args = self._extract_query_arguments(question, q_type)
        
        # Calculate confidence
        confidence = self._calculate_confidence(q_type, query_pred, focus_entities)
        
        return AnalyzedQuestion(
            original=question,
            question_type=q_type,
            expected_roles=expected_roles,
            query_predicate=query_pred,
            query_arguments=query_args,
            focus_entities=focus_entities,
            confidence=confidence,
            requires_chaining=requires_chaining
        )
    
    def _classify_question(self, question: str) -> QuestionType:
        """
        Classify the question type.
        
        Args:
            question: Lowercased question text
            
        Returns:
            QuestionType enum value
        """
        # Check for specific question words (order matters - check how_much/many before how)
        for q_type in [QuestionType.HOW_MUCH, QuestionType.HOW_MANY, QuestionType.WHO, 
                       QuestionType.WHAT, QuestionType.WHERE, QuestionType.WHEN,
                       QuestionType.WHY, QuestionType.HOW, QuestionType.WHICH]:
            patterns = self.question_patterns.get(q_type, [])
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    return q_type
        
        # Check for yes/no questions
        if question.strip().startswith(('is ', 'are ', 'was ', 'were ', 'do ', 
                                        'does ', 'did ', 'have ', 'has ', 'can ',
                                        'could ', 'will ', 'would ')):
            return QuestionType.YES_NO
        
        return QuestionType.UNKNOWN
    
    def _extract_predicate(self, question: str) -> Optional[str]:
        """
        Extract the query predicate from the question.
        
        Args:
            question: The question text
            
        Returns:
            Extracted predicate or None
        """
        for pattern in self.predicate_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        # Try to find verbs by common patterns
        verb_pattern = r'\b(\w+(?:ed|ing|es|s))\b'
        matches = re.findall(verb_pattern, question.lower())
        
        # Filter out common non-predicates
        stop_words = {'is', 'are', 'was', 'were', 'has', 'does', 'this', 'that'}
        for match in matches:
            if match not in stop_words and len(match) > 2:
                return match
        
        return None
    
    def _extract_entities(self, question: str) -> List[str]:
        """
        Extract focus entities from the question.
        
        Args:
            question: The question text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, question)
            for match in matches:
                if isinstance(match, str) and len(match) > 1:
                    # Filter out question words and common stop words
                    if match.lower() not in ('who', 'what', 'where', 'when', 
                                             'why', 'how', 'which', 'the'):
                        entities.append(match)
        
        return list(set(entities))
    
    def _requires_chaining(self, question: str) -> bool:
        """
        Determine if the question requires frame chaining.
        
        Args:
            question: Lowercased question text
            
        Returns:
            True if chaining is likely required
        """
        for pattern in self.chaining_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return True
        
        # Questions about causes or results often need chaining
        if 'cause' in question or 'result' in question or 'led to' in question:
            return True
        
        return False
    
    def _extract_query_arguments(
        self, 
        question: str, 
        q_type: QuestionType
    ) -> Dict[str, str]:
        """
        Extract arguments from the question that help locate the answer.
        
        Args:
            question: The question text
            q_type: Classified question type
            
        Returns:
            Dictionary of role: value pairs
        """
        arguments = {}
        
        # Extract temporal references
        temporal_patterns = [
            r'\b(in\s+Q[1-4])',
            r'\b(in\s+\d{4})',
            r'\b(yesterday|today|tomorrow)',
            r'\b(last\s+(?:week|month|year|quarter))',
        ]
        for pattern in temporal_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                arguments['ARGM-TMP'] = match.group(1)
                break
        
        # Extract location references
        location_patterns = [
            r'\b(in\s+(?:the\s+)?[A-Z][a-z]+)',
            r'\b(at\s+(?:the\s+)?\w+)',
        ]
        for pattern in location_patterns:
            match = re.search(pattern, question)
            if match:
                arguments['ARGM-LOC'] = match.group(1)
                break
        
        # For WHO questions, try to identify the subject of the action
        if q_type == QuestionType.WHO:
            # Check for passive constructions
            passive_match = re.search(r'\b(\w+ed)\s+by\b', question)
            if passive_match:
                arguments['predicate'] = passive_match.group(1)
        
        return arguments
    
    def _calculate_confidence(
        self, 
        q_type: QuestionType, 
        predicate: Optional[str],
        entities: List[str]
    ) -> float:
        """
        Calculate confidence score for the analysis.
        
        Args:
            q_type: Question type
            predicate: Extracted predicate
            entities: Extracted entities
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence
        
        # Known question type increases confidence
        if q_type != QuestionType.UNKNOWN:
            confidence += 0.2
        
        # Having a predicate increases confidence
        if predicate:
            confidence += 0.2
        
        # Having entities increases confidence
        if entities:
            confidence += 0.1 * min(len(entities), 2)
        
        return min(confidence, 1.0)
    
    def refine_expected_roles(
        self, 
        analysis: AnalyzedQuestion,
        context: str
    ) -> List[str]:
        """
        Refine expected roles based on context.
        
        This method implements context-aware role refinement for
        better answer extraction.
        
        Args:
            analysis: Initial question analysis
            context: Document/passage context
            
        Returns:
            Refined list of expected roles
        """
        roles = list(analysis.expected_roles)
        
        # WHO questions with passive constructions
        if analysis.question_type == QuestionType.WHO:
            if 'by' in analysis.original.lower():
                # Passive: looking for agent
                roles = ['ARG0'] + [r for r in roles if r != 'ARG0']
            else:
                # Active: might be looking for patient
                roles = ['ARG1', 'ARG0'] + [r for r in roles if r not in ['ARG0', 'ARG1']]
        
        # WHAT questions about causes
        if analysis.question_type == QuestionType.WHAT:
            if 'cause' in analysis.original.lower():
                roles = ['ARGM-CAU', 'ARG0'] + [r for r in roles if r not in ['ARGM-CAU', 'ARG0']]
        
        return roles
    
    def generate_search_query(self, analysis: AnalyzedQuestion) -> str:
        """
        Generate a search query from the analysis.
        
        Useful for finding relevant passages before SRL parsing.
        
        Args:
            analysis: Analyzed question
            
        Returns:
            Search query string
        """
        query_parts = []
        
        # Add predicate
        if analysis.query_predicate:
            query_parts.append(analysis.query_predicate)
        
        # Add entities
        query_parts.extend(analysis.focus_entities[:3])  # Limit to top 3
        
        # Add any extracted argument values
        for role, value in analysis.query_arguments.items():
            if value:
                query_parts.append(value)
        
        return ' '.join(query_parts) if query_parts else analysis.original
