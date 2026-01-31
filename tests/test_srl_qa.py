"""
Tests for SRL-QA System

This module contains unit tests for the SRL-Based Question Answering system.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from srl_qa.config import Config, SemanticRole, PredicateArgumentStructure
from srl_qa.question_analyzer import QuestionAnalyzer, QuestionType
from srl_qa.srl_parser import SRLParser
from srl_qa.answer_extractor import AnswerExtractor
from srl_qa.pipeline import SRLQAPipeline


class TestConfig(unittest.TestCase):
    """Tests for configuration module."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = Config()
        self.assertEqual(config.max_chain_depth, 3)
        self.assertEqual(config.confidence_threshold, 0.75)
        self.assertTrue(config.preserve_edited_nodes)
    
    def test_core_arguments(self):
        """Test core arguments list."""
        config = Config()
        self.assertIn('ARG0', config.core_arguments)
        self.assertIn('ARG1', config.core_arguments)
        self.assertEqual(len(config.core_arguments), 6)
    
    def test_modifier_arguments(self):
        """Test modifier arguments list."""
        config = Config()
        self.assertIn('ARGM-LOC', config.modifier_arguments)
        self.assertIn('ARGM-TMP', config.modifier_arguments)
        self.assertIn('ARGM-CAU', config.modifier_arguments)


class TestSemanticRole(unittest.TestCase):
    """Tests for SemanticRole dataclass."""
    
    def test_is_core_argument(self):
        """Test core argument detection."""
        role = SemanticRole(role='ARG0', text='test', start_idx=0, end_idx=1)
        self.assertTrue(role.is_core_argument())
        
        role = SemanticRole(role='ARGM-LOC', text='test', start_idx=0, end_idx=1)
        self.assertFalse(role.is_core_argument())
    
    def test_is_modifier(self):
        """Test modifier detection."""
        role = SemanticRole(role='ARGM-TMP', text='test', start_idx=0, end_idx=1)
        self.assertTrue(role.is_modifier())
        
        role = SemanticRole(role='ARG1', text='test', start_idx=0, end_idx=1)
        self.assertFalse(role.is_modifier())


class TestQuestionAnalyzer(unittest.TestCase):
    """Tests for question analyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.analyzer = QuestionAnalyzer(self.config)
    
    def test_who_question(self):
        """Test WHO question classification."""
        analysis = self.analyzer.analyze("Who announced the profits?")
        self.assertEqual(analysis.question_type, QuestionType.WHO)
        self.assertIn('ARG0', analysis.expected_roles)
    
    def test_what_question(self):
        """Test WHAT question classification."""
        analysis = self.analyzer.analyze("What did the company announce?")
        self.assertEqual(analysis.question_type, QuestionType.WHAT)
        self.assertIn('ARG1', analysis.expected_roles)
    
    def test_where_question(self):
        """Test WHERE question classification."""
        analysis = self.analyzer.analyze("Where did the meeting happen?")
        self.assertEqual(analysis.question_type, QuestionType.WHERE)
        self.assertIn('ARGM-LOC', analysis.expected_roles)
    
    def test_when_question(self):
        """Test WHEN question classification."""
        analysis = self.analyzer.analyze("When did they announce it?")
        self.assertEqual(analysis.question_type, QuestionType.WHEN)
        self.assertIn('ARGM-TMP', analysis.expected_roles)
    
    def test_why_question(self):
        """Test WHY question classification."""
        analysis = self.analyzer.analyze("Why did revenue decline?")
        self.assertEqual(analysis.question_type, QuestionType.WHY)
        self.assertIn('ARGM-CAU', analysis.expected_roles)
    
    def test_how_question(self):
        """Test HOW question classification."""
        analysis = self.analyzer.analyze("How did they approve the merger?")
        self.assertEqual(analysis.question_type, QuestionType.HOW)
        self.assertIn('ARGM-MNR', analysis.expected_roles)
    
    def test_predicate_extraction(self):
        """Test predicate extraction from questions."""
        analysis = self.analyzer.analyze("Who announced the profits?")
        self.assertIsNotNone(analysis.query_predicate)
        self.assertIn('announce', analysis.query_predicate.lower())


class TestSRLParser(unittest.TestCase):
    """Tests for SRL parser."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.parser = SRLParser(self.config)
    
    def test_tokenize(self):
        """Test tokenization."""
        tokens = self.parser.tokenize("The company announced profits.")
        self.assertEqual(len(tokens), 4)
        self.assertEqual(tokens[0].text, "The")
    
    def test_parse_simple_sentence(self):
        """Test parsing a simple sentence."""
        structures = self.parser.parse("The company announced record profits yesterday.")
        self.assertGreater(len(structures), 0)
        
        # Check for predicate
        predicates = [s.predicate for s in structures]
        self.assertTrue(any('announced' in p for p in predicates))
    
    def test_argument_extraction(self):
        """Test argument extraction."""
        structures = self.parser.parse("The company announced record profits yesterday.")
        
        if structures:
            structure = structures[0]
            # Should have ARG0 (agent)
            arg0 = structure.get_argument('ARG0')
            if arg0:
                self.assertIn('company', arg0.text.lower())
    
    def test_predicate_sense_assignment(self):
        """Test predicate sense assignment."""
        structures = self.parser.parse("The CEO stated the results.")
        
        if structures:
            for structure in structures:
                if 'state' in structure.predicate.lower():
                    self.assertIn('.', structure.predicate_sense)


class TestAnswerExtractor(unittest.TestCase):
    """Tests for answer extractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.parser = SRLParser(self.config)
        self.analyzer = QuestionAnalyzer(self.config)
        self.extractor = AnswerExtractor(self.config, self.parser)
    
    def test_who_extraction(self):
        """Test answer extraction for WHO questions."""
        question = "Who announced the profits?"
        passage = "The company announced record profits yesterday."
        
        analysis = self.analyzer.analyze(question)
        result = self.extractor.extract(analysis, passage)
        
        self.assertIsNotNone(result.best_answer)
        if result.best_answer:
            self.assertIn('company', result.best_answer.text.lower())
    
    def test_when_extraction(self):
        """Test answer extraction for WHEN questions."""
        question = "When did they announce it?"
        passage = "The company announced record profits yesterday."
        
        analysis = self.analyzer.analyze(question)
        result = self.extractor.extract(analysis, passage)
        
        # Should find temporal argument
        self.assertTrue(len(result.answers) > 0)
    
    def test_confidence_scoring(self):
        """Test confidence score calculation."""
        question = "Who announced the profits?"
        passage = "The company announced record profits yesterday."
        
        analysis = self.analyzer.analyze(question)
        result = self.extractor.extract(analysis, passage)
        
        if result.best_answer:
            self.assertGreaterEqual(result.best_answer.confidence, 0)
            self.assertLessEqual(result.best_answer.confidence, 1)


class TestPipeline(unittest.TestCase):
    """Tests for the main pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.pipeline = SRLQAPipeline(self.config)
    
    def test_basic_qa(self):
        """Test basic question answering."""
        question = "Who announced record profits?"
        passage = "The company announced record profits yesterday."
        
        result = self.pipeline.answer(question, passage)
        
        self.assertIsNotNone(result.answer)
        self.assertIsNotNone(result.analysis)
    
    def test_answer_with_chaining(self):
        """Test QA with frame chaining."""
        question = "What did the company announce?"
        passage = "The company announced record profits yesterday."
        
        result = self.pipeline.answer_with_chaining(question, passage)
        
        self.assertIsNotNone(result)
    
    def test_answer_with_correction(self):
        """Test QA with self-correction."""
        question = "Why did revenue decline?"
        passage = "Revenue declined sharply due to market conditions."
        
        result = self.pipeline.answer_with_correction(question, passage)
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.self_correction_result)
    
    def test_batch_processing(self):
        """Test batch question processing."""
        qa_pairs = [
            ("Who announced profits?", "The company announced record profits."),
            ("When did they announce?", "They announced it yesterday.")
        ]
        
        results = self.pipeline.answer_batch(qa_pairs)
        
        self.assertEqual(len(results), 2)
    
    def test_pipeline_info(self):
        """Test pipeline information retrieval."""
        info = self.pipeline.get_pipeline_info()
        
        self.assertIn('config', info)
        self.assertIn('components', info)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.pipeline = SRLQAPipeline(self.config)
    
    def test_who_question_integration(self):
        """Integration test for WHO questions."""
        result = self.pipeline.answer(
            "Who announced record profits?",
            "The company announced record profits yesterday."
        )
        self.assertIn('company', result.answer.lower())
    
    def test_what_question_integration(self):
        """Integration test for WHAT questions."""
        result = self.pipeline.answer(
            "What did the company announce?",
            "The company announced record profits yesterday."
        )
        # Should find something related to profits
        self.assertTrue(len(result.answer) > 0)
    
    def test_why_question_integration(self):
        """Integration test for WHY questions."""
        result = self.pipeline.answer(
            "Why did revenue decline?",
            "Revenue declined sharply due to market conditions."
        )
        # Should find the cause
        self.assertTrue(len(result.answer) > 0)
    
    def test_how_question_integration(self):
        """Integration test for HOW questions."""
        result = self.pipeline.answer(
            "How did the board approve the merger?",
            "The board approved the merger unanimously in today's meeting."
        )
        # Should find manner
        self.assertTrue(len(result.answer) > 0)


if __name__ == '__main__':
    unittest.main()
