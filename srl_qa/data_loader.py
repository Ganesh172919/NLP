"""
PropBank Data Loader Module

This module handles loading and processing PropBank annotations following
the official documentation from https://github.com/propbank/propbank-release

Key Features:
1. CoNLL format parsing (2004/2005 SRL format)
2. Frame file parsing for PropBank 3.1
3. Disfluency handling (preserves EDITED nodes per PropBank docs)
4. Support for unified frame semantics (verbal/nominal/adjectival)
5. Official train/dev/test split handling

References:
- Palmer, M., Gildea, D., & Kingsbury, P. (2005). The Proposition Bank
- PropBank 3.1 Frame Files Documentation
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

from .config import Config, PropBankFrame, SemanticRole, PredicateArgumentStructure

logger = logging.getLogger(__name__)


@dataclass
class ConLLSentence:
    """
    Represents a sentence in CoNLL format.
    
    CoNLL 2004/2005 SRL format columns:
    1. Word index
    2. Word form
    3. Lemma
    4. POS tag
    5. Parse bit (constituent parse)
    6. Predicate indicator
    7+. Argument labels (one column per predicate)
    """
    tokens: List[str]
    lemmas: List[str]
    pos_tags: List[str]
    parse_bits: List[str]
    predicates: List[Optional[str]]
    argument_labels: List[List[str]]
    sentence_id: str = ""
    
    def get_text(self) -> str:
        """Reconstruct the sentence text."""
        return " ".join(self.tokens)
    
    def get_predicate_indices(self) -> List[int]:
        """Get indices of all predicates in the sentence."""
        return [i for i, p in enumerate(self.predicates) if p is not None]


class PropBankDataLoader:
    """
    Loader for PropBank data following official repository documentation.
    
    This class handles:
    1. CoNLL format file parsing
    2. Frame file (XML) parsing for role definitions
    3. Disfluency token preservation (EDITED nodes)
    4. Train/dev/test split management
    5. Edge case handling (e.g., empty sentences)
    
    Usage:
        loader = PropBankDataLoader(config)
        train_data = loader.load_split('train')
        frames = loader.load_frames()
    """
    
    def __init__(self, config: Config):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration object with paths and settings
        """
        self.config = config
        self.frames: Dict[str, PropBankFrame] = {}
        self.unified_frames: Dict[str, List[str]] = {}  # Maps lemmas to related frames
        
    def load_conll_file(self, filepath: str) -> List[ConLLSentence]:
        """
        Load and parse a CoNLL format file.
        
        The CoNLL 2004/2005 SRL format is produced by the official
        map_all_to_conll.py script from PropBank documentation.
        
        Args:
            filepath: Path to the CoNLL file
            
        Returns:
            List of ConLLSentence objects
        """
        sentences = []
        current_sentence = self._initialize_sentence_data()
        sentence_count = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    if not line:
                        # Empty line marks sentence boundary
                        if current_sentence['tokens']:
                            sent = self._create_sentence(
                                current_sentence, 
                                f"{filepath}:{sentence_count}"
                            )
                            if sent is not None:
                                sentences.append(sent)
                            sentence_count += 1
                        current_sentence = self._initialize_sentence_data()
                        continue
                    
                    # Parse CoNLL columns
                    columns = line.split('\t')
                    if len(columns) < 6:
                        columns = line.split()
                    
                    if len(columns) >= 6:
                        self._process_conll_columns(columns, current_sentence)
                
                # Handle last sentence if file doesn't end with newline
                if current_sentence['tokens']:
                    sent = self._create_sentence(
                        current_sentence, 
                        f"{filepath}:{sentence_count}"
                    )
                    if sent is not None:
                        sentences.append(sent)
                        
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in {filepath}: {e}")
            
        return sentences
    
    def _initialize_sentence_data(self) -> Dict:
        """Initialize data structure for a new sentence."""
        return {
            'tokens': [],
            'lemmas': [],
            'pos_tags': [],
            'parse_bits': [],
            'predicates': [],
            'argument_labels': []
        }
    
    def _process_conll_columns(self, columns: List[str], data: Dict) -> None:
        """Process a single line of CoNLL data."""
        data['tokens'].append(columns[1] if len(columns) > 1 else columns[0])
        data['lemmas'].append(columns[2] if len(columns) > 2 else columns[0])
        data['pos_tags'].append(columns[3] if len(columns) > 3 else "X")
        data['parse_bits'].append(columns[4] if len(columns) > 4 else "_")
        
        # Predicate sense (column 5 or 6 depending on format)
        pred = columns[5] if len(columns) > 5 else None
        if pred in ['-', '_', '']:
            pred = None
        data['predicates'].append(pred)
        
        # Argument labels (remaining columns)
        args = columns[6:] if len(columns) > 6 else []
        data['argument_labels'].append(args)
    
    def _create_sentence(self, data: Dict, sent_id: str) -> Optional[ConLLSentence]:
        """
        Create a ConLLSentence object from parsed data.
        
        Handles the edge case mentioned in PropBank docs:
        ontonotes/nw/p2.5_c2e/00/p2.5_c2e_0034 sentence 12 (empty sentence)
        """
        # Handle empty sentence edge case
        if not data['tokens']:
            logger.debug(f"Skipping empty sentence: {sent_id}")
            return None
            
        return ConLLSentence(
            tokens=data['tokens'],
            lemmas=data['lemmas'],
            pos_tags=data['pos_tags'],
            parse_bits=data['parse_bits'],
            predicates=data['predicates'],
            argument_labels=data['argument_labels'],
            sentence_id=sent_id
        )
    
    def load_frame_file(self, filepath: str) -> List[PropBankFrame]:
        """
        Load PropBank frame definitions from XML file.
        
        PropBank 3.1 frame files contain roleset definitions for predicates,
        including unified frames for verbal, nominal, and adjectival forms.
        
        Args:
            filepath: Path to frame XML file
            
        Returns:
            List of PropBankFrame objects
        """
        frames = []
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Get predicate lemma
            predicate_elem = root.find('.//predicate')
            if predicate_elem is None:
                return frames
                
            lemma = predicate_elem.get('lemma', '')
            
            # Parse each roleset
            for roleset in root.findall('.//roleset'):
                roleset_id = roleset.get('id', '')
                sense = roleset_id.split('.')[-1] if '.' in roleset_id else '01'
                
                # Extract roles
                roles = {}
                for role in roleset.findall('.//role'):
                    arg_num = role.get('n', '')
                    descr = role.get('descr', '')
                    if arg_num:
                        arg_label = f"ARG{arg_num}" if arg_num.isdigit() else arg_num
                        roles[arg_label] = descr
                
                # Extract examples
                examples = []
                for example in roleset.findall('.//example'):
                    text_elem = example.find('text')
                    if text_elem is not None and text_elem.text:
                        examples.append(text_elem.text.strip())
                
                # Determine frame type (unified semantics)
                is_verbal = any(c.isalpha() and c.islower() for c in lemma)
                is_nominal = lemma.endswith(('tion', 'ment', 'ness', 'er', 'or'))
                is_adjectival = lemma.endswith(('ed', 'ing', 'able', 'ible'))
                
                frame = PropBankFrame(
                    predicate=lemma,
                    sense=sense,
                    roleset_id=roleset_id,
                    roles=roles,
                    examples=examples,
                    is_verbal=is_verbal,
                    is_nominal=is_nominal,
                    is_adjectival=is_adjectival
                )
                frames.append(frame)
                self.frames[roleset_id] = frame
                
        except ET.ParseError as e:
            logger.error(f"XML parse error in {filepath}: {e}")
        except FileNotFoundError:
            logger.warning(f"Frame file not found: {filepath}")
            
        return frames
    
    def load_all_frames(self, frames_dir: Optional[str] = None) -> Dict[str, PropBankFrame]:
        """
        Load all PropBank frame definitions.
        
        Args:
            frames_dir: Directory containing frame XML files
            
        Returns:
            Dictionary mapping roleset IDs to PropBankFrame objects
        """
        frames_path = frames_dir or self.config.frames_path
        
        if not os.path.exists(frames_path):
            logger.warning(f"Frames directory not found: {frames_path}")
            return self.frames
            
        for filename in os.listdir(frames_path):
            if filename.endswith('.xml'):
                filepath = os.path.join(frames_path, filename)
                self.load_frame_file(filepath)
                
        logger.info(f"Loaded {len(self.frames)} frame definitions")
        return self.frames
    
    def load_split(self, split: str = 'train') -> List[ConLLSentence]:
        """
        Load data for a specific split (train/dev/test).
        
        Uses official splits from /docs/evaluation folder per PropBank docs.
        For EWT, uses UD splits as specified in the documentation.
        
        Args:
            split: One of 'train', 'dev', or 'test'
            
        Returns:
            List of ConLLSentence objects for the split
        """
        split_path = os.path.join(self.config.test_split_path, f"{split}.txt")
        sentences = []
        
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                file_list = [line.strip() for line in f if line.strip()]
                
            for conll_file in file_list:
                full_path = os.path.join(self.config.propbank_path, conll_file)
                sentences.extend(self.load_conll_file(full_path))
        else:
            logger.warning(f"Split file not found: {split_path}")
            
        return sentences
    
    def parse_gold_skel(self, gold_skel_content: str) -> List[Dict]:
        """
        Parse .gold_skel file format (pre-conversion format).
        
        Note: Per PropBank docs, use map_all_to_conll.py for proper conversion.
        This method is for inspection/debugging only.
        
        Args:
            gold_skel_content: Content of a .gold_skel file
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        for line in gold_skel_content.strip().split('\n'):
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if len(parts) >= 6:
                annotation = {
                    'file': parts[0],
                    'sentence': int(parts[1]) if parts[1].isdigit() else 0,
                    'token': int(parts[2]) if parts[2].isdigit() else 0,
                    'predicate': parts[3],
                    'sense': parts[4],
                    'arguments': parts[5:]
                }
                annotations.append(annotation)
                
        return annotations
    
    def build_unified_frame_map(self) -> Dict[str, List[str]]:
        """
        Build a mapping of related frames for unified semantics.
        
        PropBank 3.1 unifies verbal, nominal, and adjectival predicates
        (e.g., "create" and "creation" map to the same sense).
        
        Returns:
            Dictionary mapping base lemmas to related roleset IDs
        """
        # Common derivational patterns for unified frames
        derivation_patterns = [
            (r'tion$', ''),      # creation -> create
            (r'ment$', ''),      # movement -> move
            (r'er$', ''),        # worker -> work
            (r'or$', ''),        # creator -> create
            (r'ing$', ''),       # creating -> create
            (r'ed$', ''),        # created -> create
            (r'able$', ''),      # createable -> create
            (r'ible$', ''),      # digestible -> digest
            (r'ness$', ''),      # happiness -> happy
        ]
        
        for roleset_id, frame in self.frames.items():
            lemma = frame.predicate
            base_lemma = lemma
            
            # Try to find base form
            for pattern, replacement in derivation_patterns:
                if re.search(pattern, lemma):
                    base_lemma = re.sub(pattern, replacement, lemma)
                    break
            
            if base_lemma not in self.unified_frames:
                self.unified_frames[base_lemma] = []
            self.unified_frames[base_lemma].append(roleset_id)
            
        return self.unified_frames
    
    def handle_disfluency(self, tokens: List[str], preserve: bool = True) -> List[str]:
        """
        Handle disfluency tokens per PropBank documentation.
        
        CRITICAL: Per PropBank docs, DO NOT remove EDITED nodes as was done
        in prior Ontonotes releases. This method preserves them by default.
        
        Args:
            tokens: List of tokens
            preserve: If True, preserve EDITED markers (recommended)
            
        Returns:
            Processed token list
        """
        if preserve or self.config.preserve_edited_nodes:
            return tokens
            
        # If explicitly requested to remove (NOT RECOMMENDED)
        return [t for t in tokens if not t.startswith('[EDITED')]
    
    def extract_predicate_arguments(
        self, 
        sentence: ConLLSentence
    ) -> List[PredicateArgumentStructure]:
        """
        Extract predicate-argument structures from a parsed sentence.
        
        Args:
            sentence: Parsed ConLLSentence object
            
        Returns:
            List of PredicateArgumentStructure objects
        """
        structures = []
        pred_indices = sentence.get_predicate_indices()
        
        for col_idx, pred_idx in enumerate(pred_indices):
            predicate = sentence.tokens[pred_idx]
            pred_sense = sentence.predicates[pred_idx] or f"{predicate}.01"
            
            arguments = []
            current_arg = None
            current_tokens = []
            current_start = -1
            
            for token_idx, token in enumerate(sentence.tokens):
                if col_idx < len(sentence.argument_labels[token_idx]):
                    arg_label = sentence.argument_labels[token_idx][col_idx]
                else:
                    arg_label = 'O'
                
                # Handle BIO-style labels
                if arg_label.startswith('B-'):
                    # Save previous argument
                    if current_arg is not None:
                        arguments.append(SemanticRole(
                            role=current_arg,
                            text=' '.join(current_tokens),
                            start_idx=current_start,
                            end_idx=token_idx - 1
                        ))
                    current_arg = arg_label[2:]
                    current_tokens = [token]
                    current_start = token_idx
                    
                elif arg_label.startswith('I-') and current_arg is not None:
                    current_tokens.append(token)
                    
                elif arg_label.startswith('(') and arg_label.endswith('*'):
                    # Bracket notation
                    current_arg = arg_label[1:-1]
                    current_tokens = [token]
                    current_start = token_idx
                    
                elif arg_label == '*' and current_arg is not None:
                    current_tokens.append(token)
                    
                elif arg_label.endswith(')') and current_arg is not None:
                    current_tokens.append(token)
                    arguments.append(SemanticRole(
                        role=current_arg,
                        text=' '.join(current_tokens),
                        start_idx=current_start,
                        end_idx=token_idx
                    ))
                    current_arg = None
                    current_tokens = []
                    
                else:
                    # O label or end of argument
                    if current_arg is not None:
                        arguments.append(SemanticRole(
                            role=current_arg,
                            text=' '.join(current_tokens),
                            start_idx=current_start,
                            end_idx=token_idx - 1
                        ))
                        current_arg = None
                        current_tokens = []
            
            # Handle last argument
            if current_arg is not None:
                arguments.append(SemanticRole(
                    role=current_arg,
                    text=' '.join(current_tokens),
                    start_idx=current_start,
                    end_idx=len(sentence.tokens) - 1
                ))
            
            structures.append(PredicateArgumentStructure(
                predicate=predicate,
                predicate_sense=pred_sense,
                arguments=arguments,
                sentence=sentence.get_text(),
                predicate_idx=pred_idx
            ))
            
        return structures
    
    def create_sample_data(self) -> List[Dict]:
        """
        Create sample data for testing and demonstration.
        
        Returns properly formatted sample data that mimics PropBank annotations.
        """
        sample_sentences = [
            {
                "text": "The company announced record profits yesterday.",
                "predicates": [
                    {
                        "predicate": "announced",
                        "sense": "announce.01",
                        "arguments": {
                            "ARG0": "The company",
                            "ARG1": "record profits",
                            "ARGM-TMP": "yesterday"
                        }
                    }
                ]
            },
            {
                "text": "Revenue declined sharply due to market conditions.",
                "predicates": [
                    {
                        "predicate": "declined",
                        "sense": "decline.01",
                        "arguments": {
                            "ARG1": "Revenue",
                            "ARGM-MNR": "sharply",
                            "ARGM-CAU": "market conditions"
                        }
                    }
                ]
            },
            {
                "text": "The CEO stated that the acquisition will create new opportunities.",
                "predicates": [
                    {
                        "predicate": "stated",
                        "sense": "state.01",
                        "arguments": {
                            "ARG0": "The CEO",
                            "ARG1": "that the acquisition will create new opportunities"
                        }
                    },
                    {
                        "predicate": "create",
                        "sense": "create.01",
                        "arguments": {
                            "ARG0": "the acquisition",
                            "ARG1": "new opportunities"
                        }
                    }
                ]
            },
            {
                "text": "Investors expressed concern about the delayed earnings report.",
                "predicates": [
                    {
                        "predicate": "expressed",
                        "sense": "express.01",
                        "arguments": {
                            "ARG0": "Investors",
                            "ARG1": "concern about the delayed earnings report"
                        }
                    },
                    {
                        "predicate": "delayed",
                        "sense": "delay.01",
                        "arguments": {
                            "ARG1": "earnings report"
                        }
                    }
                ]
            },
            {
                "text": "The board approved the merger unanimously in today's meeting.",
                "predicates": [
                    {
                        "predicate": "approved",
                        "sense": "approve.01",
                        "arguments": {
                            "ARG0": "The board",
                            "ARG1": "the merger",
                            "ARGM-MNR": "unanimously",
                            "ARGM-LOC": "in today's meeting"
                        }
                    }
                ]
            }
        ]
        
        return sample_sentences
