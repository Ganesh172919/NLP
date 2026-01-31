"""
SRL-QA Pipeline Module

This module provides the main pipeline that integrates all components
of the SRL-Based Question Answering system.

Pipeline Architecture:
1. Data Loading → PropBank data and frame definitions
2. Question Analysis → Classify and extract query components
3. SRL Parsing → Identify predicates and arguments
4. Answer Extraction → Map question to semantic roles
5. Frame Chaining → Multi-hop reasoning (optional)
6. Self-Correction → Iterative refinement (optional)

Usage:
    pipeline = SRLQAPipeline(config)
    result = pipeline.answer("Who announced the profits?", passage)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .config import Config, PredicateArgumentStructure
from .data_loader import PropBankDataLoader
from .srl_parser import SRLParser
from .question_analyzer import QuestionAnalyzer, AnalyzedQuestion
from .answer_extractor import AnswerExtractor, Answer, ExtractionResult
from .frame_chaining import FrameChainingQA
from .self_correcting_qa import SelfCorrectingQA, SelfCorrectingResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Complete result from the SRL-QA pipeline.
    
    Attributes:
        question: Original question
        answer: Final answer text
        confidence: Confidence score
        analysis: Question analysis
        structures: SRL structures from passage
        extraction_result: Raw extraction result
        self_correction_result: Self-correction result if enabled
        processing_time: Time taken in seconds
    """
    question: str
    answer: str = ""
    confidence: float = 0.0
    analysis: Optional[AnalyzedQuestion] = None
    structures: List[PredicateArgumentStructure] = field(default_factory=list)
    extraction_result: Optional[ExtractionResult] = None
    self_correction_result: Optional[SelfCorrectingResult] = None
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'question': self.question,
            'answer': self.answer,
            'confidence': self.confidence,
            'question_type': self.analysis.question_type.value if self.analysis else None,
            'predicate': self.analysis.query_predicate if self.analysis else None,
            'expected_roles': self.analysis.expected_roles if self.analysis else [],
            'structures_count': len(self.structures),
            'processing_time': self.processing_time
        }


class SRLQAPipeline:
    """
    Main pipeline for SRL-Based Question Answering.
    
    This class orchestrates the complete QA process from question
    to answer, integrating all system components.
    
    Features:
    - Configurable pipeline modes (simple, with_chaining, self_correcting)
    - PropBank frame integration
    - Batch processing support
    - Detailed result reporting
    
    Usage:
        config = Config()
        pipeline = SRLQAPipeline(config)
        
        # Simple QA
        result = pipeline.answer("Who announced the profits?", passage)
        
        # With frame chaining
        result = pipeline.answer_with_chaining(question, passage)
        
        # With self-correction
        result = pipeline.answer_with_correction(question, passage)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration object (uses defaults if not provided)
        """
        self.config = config or Config()
        
        # Initialize components
        self.data_loader = PropBankDataLoader(self.config)
        self.parser = SRLParser(self.config)
        self.question_analyzer = QuestionAnalyzer(self.config)
        self.answer_extractor = AnswerExtractor(self.config, self.parser)
        self.frame_chainer = FrameChainingQA(self.config)
        self.self_corrector = SelfCorrectingQA(self.config)
        
        # Load frames if available
        self.frames = {}
        self._load_frames()
    
    def _load_frames(self) -> None:
        """Load PropBank frame definitions if available."""
        try:
            self.frames = self.data_loader.load_all_frames()
            logger.info(f"Loaded {len(self.frames)} PropBank frames")
        except Exception as e:
            logger.warning(f"Could not load frames: {e}")
    
    def answer(
        self, 
        question: str, 
        passage: str,
        use_frames: bool = True
    ) -> PipelineResult:
        """
        Answer a question using the basic SRL pipeline.
        
        Args:
            question: The input question
            passage: The text passage to search
            use_frames: Whether to use PropBank frame definitions
            
        Returns:
            PipelineResult with answer and metadata
        """
        import time
        start_time = time.time()
        
        result = PipelineResult(question=question)
        
        # Step 1: Analyze question
        analysis = self.question_analyzer.analyze(question)
        result.analysis = analysis
        
        # Step 2: Parse passage
        if use_frames and self.frames:
            structures = self.parser.parse_with_frames(passage, self.frames)
        else:
            structures = self.parser.parse(passage)
        result.structures = structures
        
        # Step 3: Extract answer
        extraction = self.answer_extractor.extract(analysis, passage, structures)
        result.extraction_result = extraction
        
        # Step 4: Set final answer
        if extraction.best_answer:
            result.answer = extraction.best_answer.text
            result.confidence = extraction.best_answer.confidence
        
        result.processing_time = time.time() - start_time
        
        return result
    
    def answer_with_chaining(
        self, 
        question: str, 
        passage: str
    ) -> PipelineResult:
        """
        Answer a question using frame chaining for multi-hop reasoning.
        
        This method is particularly useful for complex questions that
        require traversing multiple predicate-argument structures.
        
        Args:
            question: The input question
            passage: The text passage
            
        Returns:
            PipelineResult with answer from chain traversal
        """
        import time
        start_time = time.time()
        
        result = PipelineResult(question=question)
        
        # Analyze question
        analysis = self.question_analyzer.analyze(question)
        result.analysis = analysis
        
        # Use frame chaining
        chain_result = self.frame_chainer.answer(analysis, passage)
        result.structures = chain_result.structures
        result.extraction_result = chain_result
        
        if chain_result.best_answer:
            result.answer = chain_result.best_answer.text
            result.confidence = chain_result.best_answer.confidence
        
        result.processing_time = time.time() - start_time
        
        return result
    
    def answer_with_correction(
        self, 
        question: str, 
        passage: str
    ) -> PipelineResult:
        """
        Answer a question using self-correcting QA workflow.
        
        This method applies iterative refinement to improve answer
        confidence through multiple correction strategies.
        
        Args:
            question: The input question
            passage: The text passage
            
        Returns:
            PipelineResult with corrected answer
        """
        import time
        start_time = time.time()
        
        result = PipelineResult(question=question)
        
        # Use self-correcting QA
        correction_result = self.self_corrector.answer(question, passage)
        result.self_correction_result = correction_result
        
        # Get analysis from corrector
        result.analysis = self.question_analyzer.analyze(question)
        
        if correction_result.extraction_result:
            result.structures = correction_result.extraction_result.structures
            result.extraction_result = correction_result.extraction_result
        
        if correction_result.final_answer:
            result.answer = correction_result.final_answer.text
            result.confidence = correction_result.confidence
        
        result.processing_time = time.time() - start_time
        
        return result
    
    def answer_batch(
        self, 
        qa_pairs: List[Tuple[str, str]],
        mode: str = "simple"
    ) -> List[PipelineResult]:
        """
        Answer multiple questions in batch.
        
        Args:
            qa_pairs: List of (question, passage) tuples
            mode: One of "simple", "chaining", or "correction"
            
        Returns:
            List of PipelineResult objects
        """
        results = []
        
        for question, passage in qa_pairs:
            if mode == "chaining":
                result = self.answer_with_chaining(question, passage)
            elif mode == "correction":
                result = self.answer_with_correction(question, passage)
            else:
                result = self.answer(question, passage)
            
            results.append(result)
        
        return results
    
    def process_document(
        self, 
        document: str,
        questions: List[str]
    ) -> List[PipelineResult]:
        """
        Answer multiple questions about a single document.
        
        Optimized version that parses the document once.
        
        Args:
            document: The document text
            questions: List of questions about the document
            
        Returns:
            List of PipelineResult objects
        """
        # Parse document once
        structures = self.parser.parse(document)
        
        results = []
        for question in questions:
            analysis = self.question_analyzer.analyze(question)
            extraction = self.answer_extractor.extract_from_structures(
                analysis, structures
            )
            
            result = PipelineResult(
                question=question,
                analysis=analysis,
                structures=structures,
                extraction_result=extraction
            )
            
            if extraction.best_answer:
                result.answer = extraction.best_answer.text
                result.confidence = extraction.best_answer.confidence
            
            results.append(result)
        
        return results
    
    def demo(self) -> None:
        """
        Run a demonstration of the SRL-QA pipeline.
        """
        print("=" * 60)
        print("SRL-Based Question Answering System Demo")
        print("=" * 60)
        
        # Sample data
        sample_data = self.data_loader.create_sample_data()
        
        demo_questions = [
            ("Who announced record profits?", 
             "The company announced record profits yesterday."),
            ("Why did revenue decline?",
             "Revenue declined sharply due to market conditions."),
            ("What will the acquisition create?",
             "The CEO stated that the acquisition will create new opportunities."),
            ("How did the board approve the merger?",
             "The board approved the merger unanimously in today's meeting."),
        ]
        
        for question, passage in demo_questions:
            print(f"\nQuestion: {question}")
            print(f"Passage: {passage}")
            
            result = self.answer(question, passage)
            
            print(f"Answer: {result.answer}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Question Type: {result.analysis.question_type.value if result.analysis else 'N/A'}")
            print(f"Expected Roles: {result.analysis.expected_roles if result.analysis else []}")
            print("-" * 40)
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline configuration."""
        return {
            'config': self.config.to_dict(),
            'frames_loaded': len(self.frames),
            'components': {
                'data_loader': type(self.data_loader).__name__,
                'parser': type(self.parser).__name__,
                'question_analyzer': type(self.question_analyzer).__name__,
                'answer_extractor': type(self.answer_extractor).__name__,
                'frame_chainer': type(self.frame_chainer).__name__,
                'self_corrector': type(self.self_corrector).__name__,
            }
        }
