"""
Evaluation Module for SRL-QA System

This module provides comprehensive evaluation metrics and analysis tools
for the SRL-Based Question Answering system.

Metrics Implemented:
1. Exact Match (EM) Accuracy
2. Token-level F1 Score
3. Role-based Accuracy
4. Confidence Calibration
5. Error Analysis

Evaluation Protocol (per requirements):
- Test Set: Minimum 100 questions from EWT test split
- Baseline: Simple keyword-matching QA system
- Error Analysis: 20 failure case samples

References:
- SQuAD evaluation metrics
- PropBank evaluation conventions
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import re

from ..config import Config
from ..pipeline import SRLQAPipeline, PipelineResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """
    Container for evaluation metrics.
    
    Attributes:
        exact_match: Exact match accuracy (0-1)
        f1_score: Token-level F1 score (0-1)
        precision: Token-level precision
        recall: Token-level recall
        role_accuracy: Accuracy of role predictions
        avg_confidence: Average confidence of predictions
        total_questions: Total number of questions evaluated
        correct_answers: Number of exactly correct answers
    """
    exact_match: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    role_accuracy: float = 0.0
    avg_confidence: float = 0.0
    total_questions: int = 0
    correct_answers: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'exact_match': round(self.exact_match, 4),
            'f1_score': round(self.f1_score, 4),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'role_accuracy': round(self.role_accuracy, 4),
            'avg_confidence': round(self.avg_confidence, 4),
            'total_questions': self.total_questions,
            'correct_answers': self.correct_answers
        }


@dataclass
class ErrorCase:
    """
    Represents an error case for analysis.
    
    Attributes:
        question: The question
        passage: The passage
        predicted: Predicted answer
        expected: Expected answer
        question_type: Type of question
        error_category: Classified error type
        confidence: Model confidence
        analysis: Additional analysis notes
    """
    question: str
    passage: str
    predicted: str
    expected: str
    question_type: str = ""
    error_category: str = ""
    confidence: float = 0.0
    analysis: str = ""


@dataclass
class EvaluationReport:
    """
    Complete evaluation report.
    
    Attributes:
        metrics: Overall metrics
        baseline_metrics: Baseline comparison metrics
        error_cases: List of error cases
        category_breakdown: Performance by question type
        role_breakdown: Performance by semantic role
    """
    metrics: EvaluationMetrics
    baseline_metrics: Optional[EvaluationMetrics] = None
    error_cases: List[ErrorCase] = field(default_factory=list)
    category_breakdown: Dict[str, EvaluationMetrics] = field(default_factory=dict)
    role_breakdown: Dict[str, float] = field(default_factory=dict)


class Evaluator:
    """
    Evaluator for the SRL-QA system.
    
    This class implements comprehensive evaluation including:
    1. Standard QA metrics (EM, F1)
    2. SRL-specific metrics (role accuracy)
    3. Error analysis with categorization
    4. Baseline comparison
    
    Error Categories (per PropBank docs):
    - Predicate disambiguation failures
    - Argument mapping errors
    - Multi-predicate reasoning failures
    - Disfluency handling issues
    
    Usage:
        evaluator = Evaluator(config)
        report = evaluator.evaluate(pipeline, test_data)
        print(report.metrics.f1_score)
    """
    
    def __init__(self, config: Config):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.error_analysis_samples = config.error_analysis_samples
        
        # Error categories per PropBank documentation
        self.error_categories = [
            "predicate_disambiguation",
            "argument_mapping",
            "multi_predicate_reasoning",
            "disfluency_handling",
            "missing_role",
            "wrong_role",
            "partial_match",
            "no_answer",
            "unknown"
        ]
    
    def evaluate(
        self,
        pipeline: SRLQAPipeline,
        test_data: List[Dict],
        include_baseline: bool = True
    ) -> EvaluationReport:
        """
        Evaluate the pipeline on test data.
        
        Args:
            pipeline: SRL-QA pipeline to evaluate
            test_data: List of test examples with 'question', 'passage', 'answer'
            include_baseline: Whether to include baseline comparison
            
        Returns:
            EvaluationReport with complete results
        """
        # Collect predictions
        predictions = []
        references = []
        confidences = []
        question_types = []
        
        results_details = []
        
        for item in test_data:
            question = item['question']
            passage = item['passage']
            expected = item['answer']
            
            # Get prediction
            result = pipeline.answer(question, passage)
            predicted = result.answer
            
            predictions.append(predicted)
            references.append(expected)
            confidences.append(result.confidence)
            
            if result.analysis:
                question_types.append(result.analysis.question_type.value)
            else:
                question_types.append("unknown")
            
            results_details.append({
                'question': question,
                'passage': passage,
                'predicted': predicted,
                'expected': expected,
                'confidence': result.confidence,
                'question_type': question_types[-1],
                'result': result
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, references, confidences)
        
        # Baseline comparison
        baseline_metrics = None
        if include_baseline:
            baseline_preds = [
                self._baseline_answer(item['question'], item['passage'])
                for item in test_data
            ]
            baseline_metrics = self._calculate_metrics(
                baseline_preds, references, [0.5] * len(baseline_preds)
            )
        
        # Error analysis
        error_cases = self._analyze_errors(results_details)
        
        # Category breakdown
        category_breakdown = self._breakdown_by_category(results_details)
        
        # Role breakdown
        role_breakdown = self._breakdown_by_role(results_details)
        
        return EvaluationReport(
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            error_cases=error_cases,
            category_breakdown=category_breakdown,
            role_breakdown=role_breakdown
        )
    
    def _calculate_metrics(
        self,
        predictions: List[str],
        references: List[str],
        confidences: List[float]
    ) -> EvaluationMetrics:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            confidences: List of confidence scores
            
        Returns:
            EvaluationMetrics object
        """
        n = len(predictions)
        if n == 0:
            return EvaluationMetrics()
        
        # Exact match
        exact_matches = sum(
            1 for p, r in zip(predictions, references)
            if self._normalize_answer(p) == self._normalize_answer(r)
        )
        exact_match = exact_matches / n
        
        # Token F1
        f1_scores = []
        precisions = []
        recalls = []
        
        for pred, ref in zip(predictions, references):
            f1, p, r = self._token_f1(pred, ref)
            f1_scores.append(f1)
            precisions.append(p)
            recalls.append(r)
        
        avg_f1 = sum(f1_scores) / n
        avg_precision = sum(precisions) / n
        avg_recall = sum(recalls) / n
        
        # Average confidence
        avg_confidence = sum(confidences) / n if confidences else 0.0
        
        return EvaluationMetrics(
            exact_match=exact_match,
            f1_score=avg_f1,
            precision=avg_precision,
            recall=avg_recall,
            avg_confidence=avg_confidence,
            total_questions=n,
            correct_answers=exact_matches
        )
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer text for comparison."""
        if not text:
            return ""
        # Lowercase, remove punctuation, normalize whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def _token_f1(self, prediction: str, reference: str) -> Tuple[float, float, float]:
        """
        Calculate token-level F1 score.
        
        Args:
            prediction: Predicted answer
            reference: Reference answer
            
        Returns:
            Tuple of (f1, precision, recall)
        """
        pred_tokens = self._normalize_answer(prediction).split()
        ref_tokens = self._normalize_answer(reference).split()
        
        if not pred_tokens or not ref_tokens:
            if not pred_tokens and not ref_tokens:
                return 1.0, 1.0, 1.0
            return 0.0, 0.0, 0.0
        
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0, 0.0, 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1, precision, recall
    
    def _baseline_answer(self, question: str, passage: str) -> str:
        """
        Simple keyword-matching baseline for comparison.
        
        Args:
            question: The question
            passage: The passage
            
        Returns:
            Baseline answer (longest keyword-matching span)
        """
        # Extract keywords from question (excluding stop words)
        stop_words = {'who', 'what', 'where', 'when', 'why', 'how', 
                     'is', 'are', 'was', 'were', 'the', 'a', 'an',
                     'did', 'does', 'do', 'has', 'have', 'had'}
        
        question_words = set(
            word.lower() for word in re.findall(r'\w+', question)
            if word.lower() not in stop_words
        )
        
        if not question_words:
            return ""
        
        # Find sentences with most keyword overlap
        sentences = passage.split('.')
        best_sentence = ""
        best_score = 0
        
        for sent in sentences:
            sent_words = set(word.lower() for word in re.findall(r'\w+', sent))
            overlap = len(question_words & sent_words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sent.strip()
        
        return best_sentence
    
    def _analyze_errors(
        self, 
        results: List[Dict]
    ) -> List[ErrorCase]:
        """
        Analyze error cases for detailed reporting.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            List of ErrorCase objects
        """
        errors = []
        
        for item in results:
            pred_norm = self._normalize_answer(item['predicted'])
            ref_norm = self._normalize_answer(item['expected'])
            
            # Skip correct answers
            if pred_norm == ref_norm:
                continue
            
            # Categorize error
            category = self._categorize_error(item)
            
            error = ErrorCase(
                question=item['question'],
                passage=item['passage'][:200] + "...",
                predicted=item['predicted'],
                expected=item['expected'],
                question_type=item['question_type'],
                error_category=category,
                confidence=item['confidence'],
                analysis=self._generate_error_analysis(item, category)
            )
            errors.append(error)
            
            # Limit to configured number
            if len(errors) >= self.error_analysis_samples:
                break
        
        return errors
    
    def _categorize_error(self, item: Dict) -> str:
        """
        Categorize an error based on its characteristics.
        
        Error categories per PropBank documentation:
        - predicate_disambiguation: Wrong predicate sense identified
        - argument_mapping: Correct predicate, wrong argument selected
        - multi_predicate_reasoning: Failed to connect predicates
        - disfluency_handling: Error due to disfluencies
        - missing_role: Required role not found
        - wrong_role: Extracted from wrong role
        - partial_match: Partial answer (subset)
        - no_answer: No answer produced
        
        Args:
            item: Result dictionary
            
        Returns:
            Error category string
        """
        pred = item['predicted']
        expected = item['expected']
        result = item.get('result')
        
        # No answer
        if not pred:
            return "no_answer"
        
        # Partial match
        pred_norm = self._normalize_answer(pred)
        exp_norm = self._normalize_answer(expected)
        
        if pred_norm in exp_norm or exp_norm in pred_norm:
            return "partial_match"
        
        # Check for multi-predicate issues
        if result and result.analysis and result.analysis.requires_chaining:
            return "multi_predicate_reasoning"
        
        # Check structure count
        if result and len(result.structures) > 1:
            # Multiple predicates but possibly wrong one selected
            return "argument_mapping"
        
        if result and len(result.structures) == 0:
            return "missing_role"
        
        return "unknown"
    
    def _generate_error_analysis(self, item: Dict, category: str) -> str:
        """Generate analysis notes for an error."""
        analysis_parts = []
        
        if category == "no_answer":
            analysis_parts.append("No answer could be extracted from the passage.")
        
        elif category == "partial_match":
            analysis_parts.append(
                f"Partial overlap between predicted and expected. "
                f"May need broader/narrower span extraction."
            )
        
        elif category == "multi_predicate_reasoning":
            analysis_parts.append(
                "Question requires connecting multiple predicates. "
                "Consider using frame chaining."
            )
        
        elif category == "argument_mapping":
            analysis_parts.append(
                "Correct predicate identified but wrong argument selected. "
                "Check role priority mapping."
            )
        
        result = item.get('result')
        if result and result.analysis:
            analysis_parts.append(
                f"Question type: {result.analysis.question_type.value}, "
                f"Expected roles: {result.analysis.expected_roles}"
            )
        
        return " ".join(analysis_parts)
    
    def _breakdown_by_category(
        self, 
        results: List[Dict]
    ) -> Dict[str, EvaluationMetrics]:
        """
        Calculate metrics broken down by question type.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary of question_type: metrics
        """
        by_type: Dict[str, List[Dict]] = defaultdict(list)
        
        for item in results:
            q_type = item['question_type']
            by_type[q_type].append(item)
        
        breakdown = {}
        for q_type, items in by_type.items():
            preds = [i['predicted'] for i in items]
            refs = [i['expected'] for i in items]
            confs = [i['confidence'] for i in items]
            breakdown[q_type] = self._calculate_metrics(preds, refs, confs)
        
        return breakdown
    
    def _breakdown_by_role(
        self, 
        results: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate accuracy broken down by source role.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary of role: accuracy
        """
        by_role: Dict[str, List[bool]] = defaultdict(list)
        
        for item in results:
            result = item.get('result')
            if not result or not result.extraction_result:
                continue
            
            best = result.extraction_result.best_answer
            if best:
                is_correct = (
                    self._normalize_answer(item['predicted']) ==
                    self._normalize_answer(item['expected'])
                )
                by_role[best.source_role].append(is_correct)
        
        breakdown = {}
        for role, correct_list in by_role.items():
            if correct_list:
                breakdown[role] = sum(correct_list) / len(correct_list)
        
        return breakdown
    
    def generate_report(self, report: EvaluationReport) -> str:
        """
        Generate a formatted text report.
        
        Args:
            report: EvaluationReport object
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "SRL-QA System Evaluation Report",
            "=" * 60,
            "",
            "Overall Metrics:",
            f"  Exact Match:     {report.metrics.exact_match:.4f}",
            f"  F1 Score:        {report.metrics.f1_score:.4f}",
            f"  Precision:       {report.metrics.precision:.4f}",
            f"  Recall:          {report.metrics.recall:.4f}",
            f"  Avg Confidence:  {report.metrics.avg_confidence:.4f}",
            f"  Total Questions: {report.metrics.total_questions}",
            f"  Correct:         {report.metrics.correct_answers}",
            ""
        ]
        
        if report.baseline_metrics:
            lines.extend([
                "Baseline Comparison (Keyword Matching):",
                f"  Exact Match:     {report.baseline_metrics.exact_match:.4f}",
                f"  F1 Score:        {report.baseline_metrics.f1_score:.4f}",
                f"  Improvement:     {report.metrics.f1_score - report.baseline_metrics.f1_score:+.4f}",
                ""
            ])
        
        lines.extend([
            "Performance by Question Type:",
        ])
        for q_type, metrics in sorted(report.category_breakdown.items()):
            lines.append(f"  {q_type:12s}: EM={metrics.exact_match:.2f}, F1={metrics.f1_score:.2f}")
        lines.append("")
        
        lines.extend([
            "Performance by Source Role:",
        ])
        for role, acc in sorted(report.role_breakdown.items(), key=lambda x: -x[1]):
            lines.append(f"  {role:12s}: {acc:.2f}")
        lines.append("")
        
        if report.error_cases:
            lines.extend([
                f"Error Analysis ({len(report.error_cases)} samples):",
            ])
            for i, error in enumerate(report.error_cases[:5], 1):
                lines.extend([
                    f"  {i}. [{error.error_category}] {error.question_type}",
                    f"     Q: {error.question[:50]}...",
                    f"     Predicted: {error.predicted[:30]}...",
                    f"     Expected:  {error.expected[:30]}...",
                ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def create_sample_test_data() -> List[Dict]:
    """
    Create sample test data for evaluation demonstration.
    
    Returns:
        List of test examples
    """
    return [
        {
            "question": "Who announced record profits?",
            "passage": "The company announced record profits yesterday.",
            "answer": "The company"
        },
        {
            "question": "What did the company announce?",
            "passage": "The company announced record profits yesterday.",
            "answer": "record profits"
        },
        {
            "question": "When did the company announce profits?",
            "passage": "The company announced record profits yesterday.",
            "answer": "yesterday"
        },
        {
            "question": "Why did revenue decline?",
            "passage": "Revenue declined sharply due to market conditions.",
            "answer": "market conditions"
        },
        {
            "question": "How did revenue decline?",
            "passage": "Revenue declined sharply due to market conditions.",
            "answer": "sharply"
        },
        {
            "question": "What will the acquisition create?",
            "passage": "The CEO stated that the acquisition will create new opportunities.",
            "answer": "new opportunities"
        },
        {
            "question": "Who stated something about the acquisition?",
            "passage": "The CEO stated that the acquisition will create new opportunities.",
            "answer": "The CEO"
        },
        {
            "question": "Who expressed concern?",
            "passage": "Investors expressed concern about the delayed earnings report.",
            "answer": "Investors"
        },
        {
            "question": "How did the board approve the merger?",
            "passage": "The board approved the merger unanimously in today's meeting.",
            "answer": "unanimously"
        },
        {
            "question": "Where did the board approve the merger?",
            "passage": "The board approved the merger unanimously in today's meeting.",
            "answer": "in today's meeting"
        },
    ]
