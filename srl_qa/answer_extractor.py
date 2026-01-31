"""
Answer Extraction Module

This module implements answer extraction from SRL-parsed text using
the analyzed question's expected semantic roles.

The extractor maps question types to PropBank argument positions:
- WHO questions → ARG0, ARG1 (agents, patients)
- WHAT questions → ARG1 (themes, objects)
- WHERE questions → ARGM-LOC (locations)
- WHEN questions → ARGM-TMP (temporal)
- WHY questions → ARGM-CAU (causes)
- HOW questions → ARGM-MNR (manner)

References:
- He et al. (2015) Question-Answer Driven Semantic Role Labeling
- Shen & Lapata (2007) Using Semantic Roles to Improve QA
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .config import Config, SemanticRole, PredicateArgumentStructure
from .question_analyzer import AnalyzedQuestion, QuestionType
from .srl_parser import SRLParser

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    """
    Represents an extracted answer.
    
    Attributes:
        text: The answer text
        source_role: The semantic role from which the answer was extracted
        source_predicate: The predicate associated with this answer
        confidence: Confidence score (0-1)
        supporting_text: The source sentence
        extraction_method: How the answer was extracted
    """
    text: str
    source_role: str
    source_predicate: str
    confidence: float = 0.0
    supporting_text: str = ""
    extraction_method: str = "role_mapping"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'answer': self.text,
            'role': self.source_role,
            'predicate': self.source_predicate,
            'confidence': self.confidence,
            'source': self.supporting_text,
            'method': self.extraction_method
        }


@dataclass
class ExtractionResult:
    """
    Complete result of answer extraction.
    
    Attributes:
        question: The original question
        answers: List of candidate answers
        best_answer: The highest-confidence answer
        structures: SRL structures from the passage
        errors: Any extraction errors
    """
    question: str
    answers: List[Answer] = field(default_factory=list)
    best_answer: Optional[Answer] = None
    structures: List[PredicateArgumentStructure] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def get_answer_text(self) -> str:
        """Get the text of the best answer or empty string."""
        return self.best_answer.text if self.best_answer else ""


class AnswerExtractor:
    """
    Extracts answers from SRL-parsed text using question analysis.
    
    The extractor implements:
    1. Role-based answer mapping
    2. Predicate matching between question and passage
    3. Multi-candidate ranking
    4. Confidence scoring
    
    Usage:
        extractor = AnswerExtractor(config)
        result = extractor.extract(question_analysis, srl_structures)
        print(result.best_answer.text)
    """
    
    def __init__(self, config: Config, parser: Optional[SRLParser] = None):
        """
        Initialize the answer extractor.
        
        Args:
            config: Configuration object
            parser: Optional SRL parser instance
        """
        self.config = config
        self.parser = parser or SRLParser(config)
        
        # Role priority for different question types
        self.role_priorities = {
            QuestionType.WHO: ['ARG0', 'ARG1', 'ARG2'],
            QuestionType.WHAT: ['ARG1', 'ARG2', 'ARG0'],
            QuestionType.WHERE: ['ARGM-LOC', 'ARG2', 'ARG4'],
            QuestionType.WHEN: ['ARGM-TMP'],
            QuestionType.WHY: ['ARGM-CAU', 'ARGM-PRP'],
            QuestionType.HOW: ['ARGM-MNR', 'ARGM-EXT'],
            QuestionType.HOW_MUCH: ['ARGM-EXT', 'ARG2'],
            QuestionType.HOW_MANY: ['ARGM-EXT', 'ARG2'],
            QuestionType.WHICH: ['ARG1', 'ARG0', 'ARG2'],
            QuestionType.YES_NO: ['ARG1', 'ARG0'],
            QuestionType.UNKNOWN: ['ARG1', 'ARG0', 'ARG2', 'ARGM-LOC', 'ARGM-TMP'],
        }
        
        # Predicate similarity mappings (for loose matching)
        self.predicate_synonyms = {
            'cause': ['result', 'lead', 'create', 'produce'],
            'announce': ['state', 'declare', 'report', 'reveal'],
            'decline': ['decrease', 'fall', 'drop', 'reduce'],
            'increase': ['rise', 'grow', 'expand', 'gain'],
            'affect': ['impact', 'influence', 'change'],
            'create': ['make', 'produce', 'generate', 'form'],
        }
    
    def extract(
        self, 
        analysis: AnalyzedQuestion,
        passage: str,
        pre_parsed: Optional[List[PredicateArgumentStructure]] = None
    ) -> ExtractionResult:
        """
        Extract answer from a passage given a question analysis.
        
        Args:
            analysis: Analyzed question
            passage: Text passage to search for answer
            pre_parsed: Optional pre-parsed SRL structures
            
        Returns:
            ExtractionResult with candidate answers
        """
        result = ExtractionResult(question=analysis.original)
        
        # Parse passage if not pre-parsed
        if pre_parsed:
            structures = pre_parsed
        else:
            structures = self.parser.parse(passage)
        
        result.structures = structures
        
        if not structures:
            result.errors.append("No predicate-argument structures found in passage")
            return result
        
        # Get role priority for this question type
        role_priority = self.role_priorities.get(
            analysis.question_type, 
            ['ARG1', 'ARG0', 'ARG2']
        )
        
        # Extract candidates
        candidates = []
        
        for structure in structures:
            # Check predicate match
            pred_match_score = self._score_predicate_match(
                structure.predicate_sense,
                analysis.query_predicate
            )
            
            # Extract from matching roles
            for role in role_priority:
                arg = structure.get_argument(role)
                if arg:
                    score = self._calculate_answer_score(
                        arg, structure, analysis, pred_match_score
                    )
                    
                    answer = Answer(
                        text=arg.text,
                        source_role=role,
                        source_predicate=structure.predicate,
                        confidence=score,
                        supporting_text=structure.sentence,
                        extraction_method="role_mapping"
                    )
                    candidates.append(answer)
        
        # Rank candidates
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        # Remove duplicates (keep highest confidence)
        seen_texts = set()
        unique_candidates = []
        for candidate in candidates:
            normalized = candidate.text.lower().strip()
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_candidates.append(candidate)
        
        result.answers = unique_candidates
        result.best_answer = unique_candidates[0] if unique_candidates else None
        
        return result
    
    def _score_predicate_match(
        self, 
        passage_pred: str, 
        question_pred: Optional[str]
    ) -> float:
        """
        Score how well a passage predicate matches the question predicate.
        
        Args:
            passage_pred: Predicate from passage (e.g., "announce.01")
            question_pred: Predicate from question (e.g., "announced")
            
        Returns:
            Match score (0-1)
        """
        if not question_pred:
            return 0.5  # Neutral if no question predicate
        
        # Extract lemma from sense
        passage_lemma = passage_pred.split('.')[0].lower() if '.' in passage_pred else passage_pred.lower()
        question_lemma = question_pred.lower().rstrip('ed').rstrip('ing').rstrip('s')
        
        # Exact match
        if passage_lemma == question_lemma or passage_lemma.startswith(question_lemma):
            return 1.0
        
        # Synonym match
        for base, synonyms in self.predicate_synonyms.items():
            if question_lemma.startswith(base) or base.startswith(question_lemma):
                if any(passage_lemma.startswith(syn) for syn in synonyms):
                    return 0.8
        
        return 0.3
    
    def _calculate_answer_score(
        self,
        argument: SemanticRole,
        structure: PredicateArgumentStructure,
        analysis: AnalyzedQuestion,
        pred_match_score: float
    ) -> float:
        """
        Calculate confidence score for an answer candidate.
        
        Args:
            argument: The semantic role argument
            structure: The predicate-argument structure
            analysis: Question analysis
            pred_match_score: Predicate match score
            
        Returns:
            Confidence score (0-1)
        """
        score = 0.0
        
        # Base score from role match
        expected_roles = analysis.expected_roles
        if argument.role in expected_roles:
            # Higher score for higher priority role
            role_idx = expected_roles.index(argument.role)
            score += 0.4 * (1 - role_idx * 0.1)
        else:
            score += 0.1
        
        # Add predicate match score
        score += 0.3 * pred_match_score
        
        # Entity matching bonus
        for entity in analysis.focus_entities:
            if entity.lower() in structure.sentence.lower():
                score += 0.1
                break
        
        # Query argument matching
        for role, value in analysis.query_arguments.items():
            if structure.has_argument(role):
                arg = structure.get_argument(role)
                if arg and value.lower() in arg.text.lower():
                    score += 0.1
        
        # Length penalty for very short or very long answers
        answer_len = len(argument.text.split())
        if answer_len < 2:
            score *= 0.9
        elif answer_len > 15:
            score *= 0.8
        
        # Use argument's own confidence if available
        score *= argument.confidence
        
        return min(score, 1.0)
    
    def extract_from_structures(
        self,
        analysis: AnalyzedQuestion,
        structures: List[PredicateArgumentStructure]
    ) -> ExtractionResult:
        """
        Extract answer from pre-parsed structures.
        
        Args:
            analysis: Analyzed question
            structures: Pre-parsed SRL structures
            
        Returns:
            ExtractionResult with candidate answers
        """
        return self.extract(analysis, "", pre_parsed=structures)
    
    def extract_all_roles(
        self,
        structures: List[PredicateArgumentStructure]
    ) -> Dict[str, List[str]]:
        """
        Extract all semantic roles from structures.
        
        Useful for debugging and analysis.
        
        Args:
            structures: SRL structures
            
        Returns:
            Dictionary mapping role labels to extracted texts
        """
        roles_dict: Dict[str, List[str]] = {}
        
        for structure in structures:
            for arg in structure.arguments:
                if arg.role not in roles_dict:
                    roles_dict[arg.role] = []
                roles_dict[arg.role].append(arg.text)
        
        return roles_dict
    
    def find_answer_by_predicate(
        self,
        structures: List[PredicateArgumentStructure],
        target_predicate: str,
        target_role: str
    ) -> Optional[str]:
        """
        Find answer for a specific predicate-role combination.
        
        Args:
            structures: SRL structures
            target_predicate: Predicate to match (lemma)
            target_role: Role to extract (e.g., 'ARG0')
            
        Returns:
            Extracted answer text or None
        """
        target_lemma = target_predicate.lower().rstrip('ed').rstrip('ing').rstrip('s')
        
        for structure in structures:
            pred_lemma = structure.predicate_sense.split('.')[0].lower() if '.' in structure.predicate_sense else structure.predicate.lower()
            
            if pred_lemma.startswith(target_lemma) or target_lemma.startswith(pred_lemma):
                arg = structure.get_argument(target_role)
                if arg:
                    return arg.text
        
        return None
    
    def rank_answers_by_context(
        self,
        candidates: List[Answer],
        context_keywords: List[str]
    ) -> List[Answer]:
        """
        Re-rank answers based on context keywords.
        
        Args:
            candidates: List of candidate answers
            context_keywords: Keywords from context
            
        Returns:
            Re-ranked answer list
        """
        for answer in candidates:
            context_score = 0
            answer_lower = answer.text.lower()
            
            for keyword in context_keywords:
                if keyword.lower() in answer_lower:
                    context_score += 0.1
            
            answer.confidence += context_score
            answer.confidence = min(answer.confidence, 1.0)
        
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        return candidates
