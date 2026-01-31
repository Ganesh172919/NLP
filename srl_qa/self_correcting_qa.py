"""
Self-Correcting QA Module

This module implements the system-level innovation layer: Self-Correcting QA
Workflow with iterative refinement based on confidence thresholds.

The system implements:
1. Initial answer extraction
2. Confidence evaluation
3. SRL re-parsing with alternate settings if confidence is low
4. Answer refinement through multiple iterations
5. Fallback strategies when confidence remains low

Workflow:
```
while confidence < threshold and iterations < max:
    answer = extract_answer(question, passage)
    if answer.confidence < threshold:
        re_parse_with_alternate_strategy()
        refine_answer()
    iterations++
return best_answer
```

References:
- Self-supervised QA systems
- Confidence calibration in NLP
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .config import Config, PredicateArgumentStructure
from .question_analyzer import QuestionAnalyzer, AnalyzedQuestion
from .srl_parser import SRLParser
from .answer_extractor import AnswerExtractor, Answer, ExtractionResult
from .frame_chaining import FrameChainingQA

logger = logging.getLogger(__name__)


class CorrectionStrategy(Enum):
    """Strategies for answer correction."""
    REPARSE = "reparse"
    ROLE_EXPANSION = "role_expansion"
    FRAME_CHAINING = "frame_chaining"
    ENTITY_MATCHING = "entity_matching"
    KEYWORD_FALLBACK = "keyword_fallback"


@dataclass
class CorrectionAttempt:
    """Record of a correction attempt."""
    iteration: int
    strategy: CorrectionStrategy
    previous_answer: Optional[str]
    new_answer: Optional[str]
    previous_confidence: float
    new_confidence: float
    success: bool


@dataclass
class SelfCorrectingResult:
    """
    Result from self-correcting QA workflow.
    
    Attributes:
        question: Original question
        final_answer: Best answer after corrections
        confidence: Final confidence score
        iterations: Number of correction iterations
        attempts: History of correction attempts
        extraction_result: Underlying extraction result
    """
    question: str
    final_answer: Optional[Answer] = None
    confidence: float = 0.0
    iterations: int = 0
    attempts: List[CorrectionAttempt] = field(default_factory=list)
    extraction_result: Optional[ExtractionResult] = None
    
    def was_corrected(self) -> bool:
        """Check if answer was corrected during the process."""
        return self.iterations > 1 and any(a.success for a in self.attempts)


class SelfCorrectingQA:
    """
    Self-Correcting QA system with iterative refinement.
    
    This system implements the workflow:
    1. Extract initial answer
    2. Evaluate confidence
    3. If confidence < threshold, apply correction strategy
    4. Repeat until threshold met or max iterations reached
    
    Correction Strategies:
    - REPARSE: Re-parse with different SRL settings
    - ROLE_EXPANSION: Look at additional semantic roles
    - FRAME_CHAINING: Use multi-hop reasoning
    - ENTITY_MATCHING: Match entities from question
    - KEYWORD_FALLBACK: Simple keyword extraction
    
    Usage:
        qa = SelfCorrectingQA(config)
        result = qa.answer("Who announced the profits?", passage)
        print(result.final_answer.text)
    """
    
    def __init__(self, config: Config):
        """
        Initialize the self-correcting QA system.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.threshold = config.confidence_threshold
        self.max_iterations = config.self_correction_iterations
        
        # Initialize components
        self.question_analyzer = QuestionAnalyzer(config)
        self.parser = SRLParser(config)
        self.extractor = AnswerExtractor(config, self.parser)
        self.frame_chainer = FrameChainingQA(config)
        
        # Strategy order (applied in sequence)
        self.strategies = [
            CorrectionStrategy.ROLE_EXPANSION,
            CorrectionStrategy.FRAME_CHAINING,
            CorrectionStrategy.ENTITY_MATCHING,
            CorrectionStrategy.KEYWORD_FALLBACK,
        ]
    
    def answer(
        self, 
        question: str, 
        passage: str
    ) -> SelfCorrectingResult:
        """
        Answer a question with self-correction.
        
        Args:
            question: The input question
            passage: The text passage to search
            
        Returns:
            SelfCorrectingResult with final answer
        """
        result = SelfCorrectingResult(question=question)
        
        # Analyze question
        analysis = self.question_analyzer.analyze(question)
        
        # Initial extraction
        extraction = self.extractor.extract(analysis, passage)
        result.extraction_result = extraction
        
        current_answer = extraction.best_answer
        current_confidence = current_answer.confidence if current_answer else 0.0
        
        result.iterations = 1
        
        # Check if correction needed
        if current_confidence >= self.threshold:
            result.final_answer = current_answer
            result.confidence = current_confidence
            return result
        
        # Apply correction strategies
        strategy_idx = 0
        while (current_confidence < self.threshold and 
               result.iterations < self.max_iterations and
               strategy_idx < len(self.strategies)):
            
            strategy = self.strategies[strategy_idx]
            
            attempt = self._apply_strategy(
                strategy,
                analysis,
                passage,
                extraction.structures,
                current_answer
            )
            attempt.iteration = result.iterations
            result.attempts.append(attempt)
            
            if attempt.success and attempt.new_confidence > current_confidence:
                current_answer = Answer(
                    text=attempt.new_answer or "",
                    source_role="corrected",
                    source_predicate="",
                    confidence=attempt.new_confidence,
                    supporting_text=passage[:100],
                    extraction_method=f"self_correction_{strategy.value}"
                )
                current_confidence = attempt.new_confidence
            
            result.iterations += 1
            strategy_idx += 1
        
        result.final_answer = current_answer
        result.confidence = current_confidence
        
        return result
    
    def _apply_strategy(
        self,
        strategy: CorrectionStrategy,
        analysis: AnalyzedQuestion,
        passage: str,
        structures: List[PredicateArgumentStructure],
        current_answer: Optional[Answer]
    ) -> CorrectionAttempt:
        """
        Apply a specific correction strategy.
        
        Args:
            strategy: The strategy to apply
            analysis: Question analysis
            passage: Text passage
            structures: SRL structures
            current_answer: Current best answer
            
        Returns:
            CorrectionAttempt with results
        """
        prev_answer = current_answer.text if current_answer else None
        prev_conf = current_answer.confidence if current_answer else 0.0
        
        new_answer = None
        new_conf = 0.0
        success = False
        
        if strategy == CorrectionStrategy.ROLE_EXPANSION:
            new_answer, new_conf = self._expand_roles(analysis, structures)
            
        elif strategy == CorrectionStrategy.FRAME_CHAINING:
            new_answer, new_conf = self._use_frame_chaining(analysis, passage)
            
        elif strategy == CorrectionStrategy.ENTITY_MATCHING:
            new_answer, new_conf = self._match_entities(analysis, structures)
            
        elif strategy == CorrectionStrategy.KEYWORD_FALLBACK:
            new_answer, new_conf = self._keyword_extraction(analysis, passage)
        
        success = new_conf > prev_conf
        
        return CorrectionAttempt(
            iteration=0,  # Set by caller
            strategy=strategy,
            previous_answer=prev_answer,
            new_answer=new_answer,
            previous_confidence=prev_conf,
            new_confidence=new_conf,
            success=success
        )
    
    def _expand_roles(
        self,
        analysis: AnalyzedQuestion,
        structures: List[PredicateArgumentStructure]
    ) -> Tuple[Optional[str], float]:
        """
        Look at additional semantic roles beyond expected ones.
        
        Args:
            analysis: Question analysis
            structures: SRL structures
            
        Returns:
            Tuple of (answer_text, confidence)
        """
        # Expand expected roles to include more options
        all_roles = [
            'ARG0', 'ARG1', 'ARG2', 'ARG3',
            'ARGM-LOC', 'ARGM-TMP', 'ARGM-MNR', 'ARGM-CAU'
        ]
        
        best_answer = None
        best_conf = 0.0
        
        for structure in structures:
            for role in all_roles:
                arg = structure.get_argument(role)
                if arg:
                    # Score based on how well it matches expected roles
                    conf = 0.5
                    if role in analysis.expected_roles:
                        conf += 0.2
                    
                    # Entity overlap bonus
                    for entity in analysis.focus_entities:
                        if entity.lower() in structure.sentence.lower():
                            conf += 0.1
                    
                    if conf > best_conf:
                        best_answer = arg.text
                        best_conf = conf
        
        return best_answer, best_conf
    
    def _use_frame_chaining(
        self,
        analysis: AnalyzedQuestion,
        passage: str
    ) -> Tuple[Optional[str], float]:
        """
        Apply frame chaining for multi-hop reasoning.
        
        Args:
            analysis: Question analysis
            passage: Text passage
            
        Returns:
            Tuple of (answer_text, confidence)
        """
        result = self.frame_chainer.answer(analysis, passage)
        
        if result.best_answer:
            return result.best_answer.text, result.best_answer.confidence
        
        return None, 0.0
    
    def _match_entities(
        self,
        analysis: AnalyzedQuestion,
        structures: List[PredicateArgumentStructure]
    ) -> Tuple[Optional[str], float]:
        """
        Match entities from question to passage.
        
        Args:
            analysis: Question analysis
            structures: SRL structures
            
        Returns:
            Tuple of (answer_text, confidence)
        """
        if not analysis.focus_entities:
            return None, 0.0
        
        best_match = None
        best_conf = 0.0
        
        for structure in structures:
            for arg in structure.arguments:
                # Check entity overlap
                entity_overlap = 0
                for entity in analysis.focus_entities:
                    if entity.lower() in arg.text.lower():
                        entity_overlap += 1
                
                if entity_overlap > 0:
                    conf = 0.4 + (0.1 * entity_overlap)
                    if conf > best_conf:
                        best_match = arg.text
                        best_conf = conf
        
        return best_match, best_conf
    
    def _keyword_extraction(
        self,
        analysis: AnalyzedQuestion,
        passage: str
    ) -> Tuple[Optional[str], float]:
        """
        Fallback keyword-based extraction.
        
        Args:
            analysis: Question analysis
            passage: Text passage
            
        Returns:
            Tuple of (answer_text, confidence)
        """
        import re
        
        # This is a simple fallback - not ideal but provides an answer
        sentences = passage.split('.')
        
        best_sentence = None
        best_score = 0
        
        # Score sentences by keyword overlap
        keywords = set()
        if analysis.query_predicate:
            keywords.add(analysis.query_predicate.lower())
        for entity in analysis.focus_entities:
            keywords.update(entity.lower().split())
        
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for kw in keywords if kw in sent_lower)
            if score > best_score:
                best_score = score
                best_sentence = sent.strip()
        
        if best_sentence:
            # Try to extract a specific span
            # For WHO questions, look for noun phrases
            if analysis.question_type.value == "who":
                match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', best_sentence)
                if match:
                    return match.group(1), 0.4
            
            # For other questions, return the sentence
            return best_sentence, 0.3
        
        return None, 0.0
    
    def get_correction_report(self, result: SelfCorrectingResult) -> str:
        """
        Generate a report of the correction process.
        
        Args:
            result: Self-correcting result
            
        Returns:
            Formatted report string
        """
        lines = [
            f"Question: {result.question}",
            f"Final Answer: {result.final_answer.text if result.final_answer else 'None'}",
            f"Confidence: {result.confidence:.2f}",
            f"Iterations: {result.iterations}",
            "",
            "Correction Attempts:"
        ]
        
        for attempt in result.attempts:
            lines.append(
                f"  {attempt.iteration}. {attempt.strategy.value}: "
                f"{attempt.previous_confidence:.2f} → {attempt.new_confidence:.2f} "
                f"({'Success' if attempt.success else 'No improvement'})"
            )
        
        return "\n".join(lines)
