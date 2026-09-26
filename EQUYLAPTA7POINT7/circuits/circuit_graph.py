"""circuits/circuit_graph.py

Defines data structures for representing a Functional Dependency Circuit (FDC)
as a directed computational graph of neural components and dependency edges.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class FDCNode:
    node_id: str
    component_type: str       # 'attention_head', 'mlp_block', 'residual_subspace'
    layer: int
    head_index: Optional[int] = None
    parameter_count: int = 0
    role: str = ""             # 'core_specialist', 'immediate_consumer', 'routing_node', 'execution_node'
    downstream_targets: List[str] = field(default_factory=list)


@dataclass
class FDCEdge:
    source_node: str
    target_node: str
    edge_type: str            # 'residual_flow', 'direct_mlp_projection', 'inter_layer_attention'
    causal_weight: float = 1.0


@dataclass
class FDCGraph:
    graph_id: str
    domain: str
    donor_model: str
    donor_family: str
    core_component_id: str
    nodes: Dict[str, FDCNode] = field(default_factory=dict)
    edges: List[FDCEdge] = field(default_factory=list)

    def add_node(self, node: FDCNode):
        self.nodes[node.node_id] = node

    def add_edge(self, source: str, target: str, edge_type: str, weight: float = 1.0):
        self.edges.append(FDCEdge(source, target, edge_type, weight))
        if source in self.nodes:
            if target not in self.nodes[source].downstream_targets:
                self.nodes[source].downstream_targets.append(target)

    def total_parameters(self) -> int:
        return sum(n.parameter_count for n in self.nodes.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "domain": self.domain,
            "donor_model": self.donor_model,
            "donor_family": self.donor_family,
            "core_component_id": self.core_component_id,
            "total_nodes": len(self.nodes),
            "total_parameters": self.total_parameters(),
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
            "edges": [asdict(e) for e in self.edges]
        }


def build_math_fdc() -> FDCGraph:
    """Builds the canonical Functional Dependency Circuit for Mathematics in e4-math-4L."""
    g = FDCGraph(
        graph_id="FDC-MATH-01",
        domain="math",
        donor_model="e4-math-4L",
        donor_family="Family A (Micro-GPT)",
        core_component_id="L0_head_2"
    )
    # 1. Core Component
    g.add_node(FDCNode(
        node_id="L0_head_2",
        component_type="attention_head",
        layer=0,
        head_index=2,
        parameter_count=1024,
        role="core_specialist"
    ))
    # 2. Immediate Downstream MLP
    g.add_node(FDCNode(
        node_id="L0_mlp",
        component_type="mlp_block",
        layer=0,
        head_index=None,
        parameter_count=32768,
        role="immediate_consumer"
    ))
    # 3. Routing Head in Layer 1
    g.add_node(FDCNode(
        node_id="L1_head_1",
        component_type="attention_head",
        layer=1,
        head_index=1,
        parameter_count=1024,
        role="routing_node"
    ))
    # 4. Execution Head in Layer 2
    g.add_node(FDCNode(
        node_id="L2_head_3",
        component_type="attention_head",
        layer=2,
        head_index=3,
        parameter_count=1024,
        role="execution_node"
    ))

    # Edges
    g.add_edge("L0_head_2", "L0_mlp", "direct_mlp_projection", weight=0.68)
    g.add_edge("L0_mlp", "L1_head_1", "residual_flow", weight=0.55)
    g.add_edge("L1_head_1", "L2_head_3", "inter_layer_attention", weight=0.72)

    return g
