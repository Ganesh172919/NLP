#!/usr/bin/env python3
"""
SRL-Based Question Answering System with PropBank

Main entry point for the SRL-QA system. This script provides:
1. Interactive QA mode
2. Batch evaluation mode
3. Demo mode

Usage:
    python main.py --demo                          # Run demonstration
    python main.py --interactive                   # Interactive QA mode
    python main.py --evaluate test_data.json       # Evaluate on test data

Author: SRL-QA Research Team
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from srl_qa import (
    Config,
    SRLQAPipeline,
    QuestionAnalyzer,
    SRLParser,
    AnswerExtractor,
    FrameChainingQA,
    SelfCorrectingQA
)
from srl_qa.evaluation.metrics import Evaluator, create_sample_test_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_demo(config: Config) -> None:
    """
    Run a demonstration of the SRL-QA system.
    
    Args:
        config: Configuration object
    """
    print("=" * 70)
    print("SRL-Based Question Answering System with PropBank")
    print("Demonstration Mode")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = SRLQAPipeline(config)
    
    # Demo questions and passages
    demo_data = [
        {
            "question": "Who announced record profits?",
            "passage": "The company announced record profits yesterday."
        },
        {
            "question": "Why did revenue decline?",
            "passage": "Revenue declined sharply due to market conditions."
        },
        {
            "question": "What will the acquisition create?",
            "passage": "The CEO stated that the acquisition will create new opportunities."
        },
        {
            "question": "How did the board approve the merger?",
            "passage": "The board approved the merger unanimously in today's meeting."
        },
        {
            "question": "When did investors express concern?",
            "passage": "Investors expressed concern yesterday about the delayed earnings report."
        }
    ]
    
    print("\n" + "-" * 70)
    print("Basic SRL-QA Pipeline")
    print("-" * 70)
    
    for item in demo_data:
        question = item["question"]
        passage = item["passage"]
        
        print(f"\nQuestion: {question}")
        print(f"Passage:  {passage}")
        
        # Basic pipeline
        result = pipeline.answer(question, passage)
        
        print(f"Answer:   {result.answer}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Question Type: {result.analysis.question_type.value if result.analysis else 'N/A'}")
        print(f"Expected Roles: {result.analysis.expected_roles if result.analysis else []}")
    
    print("\n" + "-" * 70)
    print("Frame Chaining Demo (Multi-hop Reasoning)")
    print("-" * 70)
    
    # Multi-hop example
    complex_passage = (
        "The acquisition was announced by the CEO last week. "
        "This acquisition will significantly impact our market position. "
        "The market position improvement will create new job opportunities."
    )
    
    chain_question = "What will the acquisition create?"
    print(f"\nQuestion: {chain_question}")
    print(f"Passage:  {complex_passage[:100]}...")
    
    chain_result = pipeline.answer_with_chaining(chain_question, complex_passage)
    print(f"Answer:   {chain_result.answer}")
    print(f"Confidence: {chain_result.confidence:.2f}")
    
    print("\n" + "-" * 70)
    print("Self-Correcting QA Demo")
    print("-" * 70)
    
    correction_question = "What caused the market decline?"
    correction_passage = "Various economic factors caused the market decline in Q3."
    
    print(f"\nQuestion: {correction_question}")
    print(f"Passage:  {correction_passage}")
    
    correction_result = pipeline.answer_with_correction(correction_question, correction_passage)
    print(f"Answer:   {correction_result.answer}")
    print(f"Confidence: {correction_result.confidence:.2f}")
    
    if correction_result.self_correction_result:
        print(f"Iterations: {correction_result.self_correction_result.iterations}")
        print(f"Was Corrected: {correction_result.self_correction_result.was_corrected()}")
    
    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)


def run_interactive(config: Config) -> None:
    """
    Run interactive QA mode.
    
    Args:
        config: Configuration object
    """
    print("=" * 70)
    print("SRL-Based Question Answering System - Interactive Mode")
    print("=" * 70)
    print("\nCommands:")
    print("  /passage <text>  - Set the passage to search")
    print("  /mode <mode>     - Set mode (simple, chaining, correction)")
    print("  /info            - Show current settings")
    print("  /quit            - Exit")
    print("\nOr just type a question to get an answer.")
    print("-" * 70)
    
    pipeline = SRLQAPipeline(config)
    current_passage = ""
    current_mode = "simple"
    
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.startswith("/passage "):
            current_passage = user_input[9:].strip()
            print(f"Passage set ({len(current_passage)} chars)")
            continue
        
        if user_input.startswith("/mode "):
            mode = user_input[6:].strip()
            if mode in ["simple", "chaining", "correction"]:
                current_mode = mode
                print(f"Mode set to: {current_mode}")
            else:
                print("Invalid mode. Use: simple, chaining, or correction")
            continue
        
        if user_input == "/info":
            print(f"Current passage: {current_passage[:50]}..." if current_passage else "No passage set")
            print(f"Current mode: {current_mode}")
            continue
        
        if user_input == "/quit":
            print("Goodbye!")
            break
        
        # Process as question
        if not current_passage:
            print("Please set a passage first with /passage <text>")
            continue
        
        question = user_input
        
        if current_mode == "chaining":
            result = pipeline.answer_with_chaining(question, current_passage)
        elif current_mode == "correction":
            result = pipeline.answer_with_correction(question, current_passage)
        else:
            result = pipeline.answer(question, current_passage)
        
        print(f"\nAnswer: {result.answer}")
        print(f"Confidence: {result.confidence:.2f}")
        
        if result.analysis:
            print(f"Question Type: {result.analysis.question_type.value}")


def run_evaluation(config: Config, test_file: str) -> None:
    """
    Run evaluation on test data.
    
    Args:
        config: Configuration object
        test_file: Path to test data JSON file
    """
    print("=" * 70)
    print("SRL-Based Question Answering System - Evaluation Mode")
    print("=" * 70)
    
    # Load test data
    if test_file and Path(test_file).exists():
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} test examples from {test_file}")
    else:
        print("Using sample test data (10 examples)")
        test_data = create_sample_test_data()
    
    # Initialize components
    pipeline = SRLQAPipeline(config)
    evaluator = Evaluator(config)
    
    # Run evaluation
    print("\nRunning evaluation...")
    report = evaluator.evaluate(pipeline, test_data)
    
    # Print report
    print("\n" + evaluator.generate_report(report))
    
    # Save report
    output_path = Path(config.output_path) / "evaluation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_dict = {
        'metrics': report.metrics.to_dict(),
        'baseline_metrics': report.baseline_metrics.to_dict() if report.baseline_metrics else None,
        'category_breakdown': {
            k: v.to_dict() for k, v in report.category_breakdown.items()
        },
        'role_breakdown': report.role_breakdown,
        'error_count': len(report.error_cases)
    }
    
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    print(f"\nReport saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SRL-Based Question Answering System with PropBank"
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Run demonstration mode"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run interactive QA mode"
    )
    
    parser.add_argument(
        "--evaluate", "-e",
        type=str,
        nargs="?",
        const="",
        help="Evaluate on test data (provide JSON file path or use sample data)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration JSON file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        config = Config()
    
    config.output_path = args.output
    
    # Run appropriate mode
    if args.demo:
        run_demo(config)
    elif args.interactive:
        run_interactive(config)
    elif args.evaluate is not None:
        run_evaluation(config, args.evaluate)
    else:
        # Default to demo
        run_demo(config)


if __name__ == "__main__":
    main()
