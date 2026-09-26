"""circuits/circuit_graph.py

Data structures for causal circuits and graphs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class CircuitNode:
    id: str
    role: str
    layer: int
    params: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitEdge:
    source: str
    target: str
    edge_type: str
    weight: float
    causally_verified: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalCircuit:
    circuit_id: str
    model_name: str
    nodes: List[CircuitNode]
    edges: List[CircuitEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def total_parameters(self) -> int:
        return sum(n.params for n in self.nodes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.circuit_id,
            "model": self.model_name,
            "total_parameters": self.total_parameters(),
            "nodes": [
                {"id": n.id, "role": n.role, "layer": n.layer, "params": n.params, **n.metadata}
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "edge_type": e.edge_type,
                    "weight": e.weight,
                    "causally_verified": e.causally_verified,
                    **e.metadata
                }
                for e in self.edges
            ],
            "metadata": self.metadata
        }
