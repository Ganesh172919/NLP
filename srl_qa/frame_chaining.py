"""
Frame-Chaining QA Module

This module implements the algorithmic innovation layer: Frame-Chaining QA
Architecture for multi-hop reasoning across predicate-argument structures.

Frame chaining enables answering complex questions that require traversing
multiple predicates, such as:
- "What caused the revenue decline?" → Find cause predicate → trace to effect
- "Who created the policy that affected sales?" → Chain create → affect

Architecture:
1. Build predicate-argument graph from document
2. Identify question entry point
3. Traverse edges following role connections
4. Aggregate answer from chain terminus

References:
- PropBank Unified Frames for cross-predicate reasoning
- AMR-PropBank alignment for graph structures
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .config import Config, SemanticRole, PredicateArgumentStructure
from .question_analyzer import AnalyzedQuestion, QuestionType
from .srl_parser import SRLParser
from .answer_extractor import Answer, ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class FrameNode:
    """
    Node in the frame chain graph representing a predicate.
    
    Attributes:
        predicate: The predicate word
        sense: PropBank sense (e.g., "cause.01")
        arguments: Dictionary of role: text mappings
        sentence_idx: Index of source sentence
        position: Position in sentence
    """
    predicate: str
    sense: str
    arguments: Dict[str, str]
    sentence_idx: int = 0
    position: int = 0
    
    def __hash__(self):
        return hash((self.predicate, self.sense, self.sentence_idx, self.position))
    
    def __eq__(self, other):
        if not isinstance(other, FrameNode):
            return False
        return (self.predicate == other.predicate and 
                self.sense == other.sense and
                self.sentence_idx == other.sentence_idx)


@dataclass
class FrameEdge:
    """
    Edge connecting two frames through shared arguments.
    
    Attributes:
        source: Source frame node
        target: Target frame node
        shared_arg: The argument text that connects them
        source_role: Role in source frame
        target_role: Role in target frame
        weight: Connection strength
    """
    source: FrameNode
    target: FrameNode
    shared_arg: str
    source_role: str
    target_role: str
    weight: float = 1.0


@dataclass
class ChainPath:
    """
    A path through the frame graph.
    
    Attributes:
        nodes: Ordered list of frame nodes
        edges: Edges connecting the nodes
        answer_node: The terminal node containing the answer
        answer_role: The role containing the answer
        confidence: Path confidence score
    """
    nodes: List[FrameNode]
    edges: List[FrameEdge]
    answer_node: Optional[FrameNode] = None
    answer_role: Optional[str] = None
    confidence: float = 0.0
    
    def get_answer(self) -> Optional[str]:
        """Get the answer from the chain terminus."""
        if self.answer_node and self.answer_role:
            return self.answer_node.arguments.get(self.answer_role)
        return None


class FrameChainingQA:
    """
    Frame-Chaining QA system for multi-hop reasoning.
    
    This class implements the algorithmic innovation layer that enables
    answering complex questions by traversing predicate-argument structures.
    
    Algorithm (pseudocode):
    ```
    function ChainFrames(question, document):
        G = BuildPredicateGraph(document)
        entry = FindEntryPredicate(question, G)
        target_role = MapQuestionToRole(question)
        
        for path in BFS(G, entry, max_depth=3):
            if path.terminus has target_role:
                candidates.add(ExtractAnswer(path))
        
        return RankByConfidence(candidates)
    ```
    
    Usage:
        chainer = FrameChainingQA(config)
        result = chainer.answer(question_analysis, document)
    """
    
    def __init__(self, config: Config):
        """
        Initialize the frame chaining system.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.parser = SRLParser(config)
        self.max_depth = config.max_chain_depth
        
        # Causal predicate patterns for chain initiation
        self.causal_predicates = {
            'cause', 'result', 'lead', 'create', 'produce',
            'affect', 'impact', 'influence', 'trigger', 'spark'
        }
        
        # Role connection patterns (how arguments link frames)
        self.role_connections = {
            ('ARG1', 'ARG0'): 1.0,  # Theme becomes agent
            ('ARG1', 'ARG1'): 0.9,  # Theme to theme
            ('ARG0', 'ARG1'): 0.8,  # Agent becomes theme
            ('ARG2', 'ARG1'): 0.7,  # Indirect to theme
        }
    
    def build_frame_graph(
        self, 
        structures: List[PredicateArgumentStructure]
    ) -> Tuple[List[FrameNode], List[FrameEdge]]:
        """
        Build a graph of connected predicate frames.
        
        Args:
            structures: List of predicate-argument structures
            
        Returns:
            Tuple of (nodes, edges)
        """
        nodes = []
        edges = []
        
        # Create nodes
        for idx, structure in enumerate(structures):
            node = FrameNode(
                predicate=structure.predicate,
                sense=structure.predicate_sense,
                arguments={arg.role: arg.text for arg in structure.arguments},
                sentence_idx=idx,
                position=structure.predicate_idx
            )
            nodes.append(node)
        
        # Create edges based on shared arguments
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i >= j:  # Avoid duplicates and self-loops
                    continue
                
                # Check for shared arguments
                for role1, text1 in node1.arguments.items():
                    for role2, text2 in node2.arguments.items():
                        similarity = self._argument_similarity(text1, text2)
                        if similarity > 0.5:
                            weight = self.role_connections.get(
                                (role1, role2), 
                                0.5
                            ) * similarity
                            
                            edge = FrameEdge(
                                source=node1,
                                target=node2,
                                shared_arg=text1,
                                source_role=role1,
                                target_role=role2,
                                weight=weight
                            )
                            edges.append(edge)
        
        return nodes, edges
    
    def _argument_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two argument texts.
        
        Args:
            text1: First argument text
            text2: Second argument text
            
        Returns:
            Similarity score (0-1)
        """
        # Exact match
        if text1.lower().strip() == text2.lower().strip():
            return 1.0
        
        # One contains the other
        t1_lower = text1.lower()
        t2_lower = text2.lower()
        if t1_lower in t2_lower or t2_lower in t1_lower:
            return 0.8
        
        # Word overlap
        words1 = set(t1_lower.split())
        words2 = set(t2_lower.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union)
        return jaccard
    
    def find_entry_node(
        self, 
        analysis: AnalyzedQuestion,
        nodes: List[FrameNode]
    ) -> Optional[FrameNode]:
        """
        Find the entry point for chain traversal.
        
        Args:
            analysis: Question analysis
            nodes: Frame graph nodes
            
        Returns:
            Entry node or None
        """
        if not nodes:
            return None
        
        best_node = None
        best_score = 0.0
        
        query_pred = analysis.query_predicate or ""
        query_pred_lemma = query_pred.lower().rstrip('ed').rstrip('ing').rstrip('s')
        
        for node in nodes:
            score = 0.0
            node_lemma = node.predicate.lower().rstrip('ed').rstrip('ing').rstrip('s')
            
            # Predicate match
            if node_lemma == query_pred_lemma or node_lemma.startswith(query_pred_lemma):
                score += 0.5
            elif query_pred_lemma.startswith(node_lemma):
                score += 0.3
            
            # Entity match in arguments
            for entity in analysis.focus_entities:
                for arg_text in node.arguments.values():
                    if entity.lower() in arg_text.lower():
                        score += 0.3
                        break
            
            # Causal predicate bonus for WHY questions
            if analysis.question_type == QuestionType.WHY:
                if node_lemma in self.causal_predicates:
                    score += 0.2
            
            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node
    
    def traverse_chain(
        self,
        entry: FrameNode,
        nodes: List[FrameNode],
        edges: List[FrameEdge],
        target_roles: List[str],
        max_depth: int = 3
    ) -> List[ChainPath]:
        """
        Traverse the frame graph to find answer paths.
        
        Implements BFS traversal with depth limit.
        
        Args:
            entry: Starting node
            nodes: All graph nodes
            edges: All graph edges
            target_roles: Roles that might contain the answer
            max_depth: Maximum traversal depth
            
        Returns:
            List of candidate paths
        """
        paths = []
        
        # Build adjacency structure
        adjacency: Dict[FrameNode, List[FrameEdge]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.source].append(edge)
            # Add reverse edges
            reverse_edge = FrameEdge(
                source=edge.target,
                target=edge.source,
                shared_arg=edge.shared_arg,
                source_role=edge.target_role,
                target_role=edge.source_role,
                weight=edge.weight * 0.8  # Slight penalty for reverse
            )
            adjacency[edge.target].append(reverse_edge)
        
        # BFS with path tracking
        queue = [(entry, [entry], [], 0, 1.0)]  # (node, path_nodes, path_edges, depth, conf)
        visited: Set[FrameNode] = set()
        
        while queue:
            current, path_nodes, path_edges, depth, conf = queue.pop(0)
            
            if depth > max_depth:
                continue
            
            # Check if current node has target role
            for role in target_roles:
                if role in current.arguments:
                    path = ChainPath(
                        nodes=path_nodes.copy(),
                        edges=path_edges.copy(),
                        answer_node=current,
                        answer_role=role,
                        confidence=conf
                    )
                    paths.append(path)
            
            visited.add(current)
            
            # Expand neighbors
            for edge in adjacency[current]:
                if edge.target not in visited:
                    new_conf = conf * edge.weight
                    if new_conf > 0.1:  # Prune low-confidence paths
                        queue.append((
                            edge.target,
                            path_nodes + [edge.target],
                            path_edges + [edge],
                            depth + 1,
                            new_conf
                        ))
        
        return paths
    
    def answer(
        self,
        analysis: AnalyzedQuestion,
        document: str
    ) -> ExtractionResult:
        """
        Answer a question using frame chaining.
        
        Args:
            analysis: Analyzed question
            document: Document text
            
        Returns:
            ExtractionResult with answers
        """
        result = ExtractionResult(question=analysis.original)
        
        # Parse document
        sentences = document.split('.')
        all_structures = []
        for sent in sentences:
            sent = sent.strip()
            if sent:
                structures = self.parser.parse(sent + '.')
                all_structures.extend(structures)
        
        result.structures = all_structures
        
        if not all_structures:
            result.errors.append("No structures found in document")
            return result
        
        # Build frame graph
        nodes, edges = self.build_frame_graph(all_structures)
        
        # Find entry point
        entry = self.find_entry_node(analysis, nodes)
        
        if not entry:
            # Fallback: use first node
            entry = nodes[0] if nodes else None
        
        if not entry:
            result.errors.append("Could not find entry point for chain traversal")
            return result
        
        # Traverse to find answers
        target_roles = analysis.expected_roles
        paths = self.traverse_chain(entry, nodes, edges, target_roles, self.max_depth)
        
        # Convert paths to answers
        for path in paths:
            answer_text = path.get_answer()
            if answer_text:
                answer = Answer(
                    text=answer_text,
                    source_role=path.answer_role or "",
                    source_predicate=path.answer_node.predicate if path.answer_node else "",
                    confidence=path.confidence,
                    supporting_text=self._reconstruct_path_text(path),
                    extraction_method="frame_chaining"
                )
                result.answers.append(answer)
        
        # Sort by confidence
        result.answers.sort(key=lambda x: x.confidence, reverse=True)
        
        # Set best answer
        if result.answers:
            result.best_answer = result.answers[0]
        
        return result
    
    def _reconstruct_path_text(self, path: ChainPath) -> str:
        """Reconstruct the reasoning path as text."""
        if not path.nodes:
            return ""
        
        parts = []
        for node in path.nodes:
            arg_str = ", ".join(f"{k}={v}" for k, v in node.arguments.items())
            parts.append(f"{node.predicate}({arg_str})")
        
        return " → ".join(parts)
    
    def explain_chain(self, path: ChainPath) -> str:
        """
        Generate human-readable explanation of the chain.
        
        Args:
            path: Chain path
            
        Returns:
            Explanation string
        """
        if not path.nodes:
            return "Empty chain"
        
        explanations = []
        
        for i, node in enumerate(path.nodes):
            if i == 0:
                explanations.append(f"Starting from predicate '{node.predicate}'")
            else:
                if i - 1 < len(path.edges):
                    edge = path.edges[i - 1]
                    explanations.append(
                        f"Following shared argument '{edge.shared_arg}' "
                        f"({edge.source_role} → {edge.target_role}) "
                        f"to predicate '{node.predicate}'"
                    )
        
        if path.answer_node and path.answer_role:
            explanations.append(
                f"Found answer in {path.answer_role}: "
                f"'{path.answer_node.arguments.get(path.answer_role, 'N/A')}'"
            )
        
        return "\n".join(explanations)
