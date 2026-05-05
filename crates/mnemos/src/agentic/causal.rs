//! Causal Memory - Hypergraph Relationships
//!
//! Implements a causal knowledge graph where agents can:
//! 1. Record cause-effect relationships
//! 2. Build hypergraph structures (edges connecting multiple nodes)
//! 3. Query causal chains and relationships
//! 4. Reason about consequences of actions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// A node in the causal graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalNode {
    /// Unique identifier.
    pub id: Uuid,

    /// Node label/name.
    pub label: String,

    /// Node type/category.
    pub node_type: String,

    /// Description.
    pub description: String,

    /// Embedding for similarity search.
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,

    /// When this node was created.
    pub created_at: DateTime<Utc>,
}

impl CausalNode {
    /// Create a new causal node.
    pub fn new(label: impl Into<String>, node_type: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            label: label.into(),
            node_type: node_type.into(),
            description: String::new(),
            embedding: None,
            created_at: Utc::now(),
        }
    }

    /// Add description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Add embedding.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// A directed edge representing causation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdge {
    /// Unique identifier.
    pub id: Uuid,

    /// Source node (cause).
    pub cause: Uuid,

    /// Target node (effect).
    pub effect: Uuid,

    /// Relationship type (e.g., "causes", "prevents", "enables").
    pub relationship: String,

    /// Strength of the causal relationship (0.0 to 1.0).
    pub strength: f32,

    /// Number of observations supporting this edge.
    pub evidence_count: u32,
}

impl CausalEdge {
    /// Create a new causal edge.
    pub fn new(cause: Uuid, effect: Uuid, relationship: impl Into<String>, strength: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            cause,
            effect,
            relationship: relationship.into(),
            strength: strength.clamp(0.0, 1.0),
            evidence_count: 1,
        }
    }

    /// Add evidence (increases count and adjusts strength).
    pub fn add_evidence(&mut self, observed_strength: f32) {
        self.evidence_count += 1;
        // Bayesian update of strength
        let n = self.evidence_count as f32;
        self.strength = ((n - 1.0) * self.strength + observed_strength) / n;
    }
}

/// A hyperedge connecting multiple nodes.
///
/// Unlike regular edges which connect two nodes, hyperedges can
/// represent complex relationships like "A AND B cause C".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperedge {
    /// Unique identifier.
    pub id: Uuid,

    /// Source nodes (all must be present for the effect).
    pub causes: Vec<Uuid>,

    /// Target nodes (effects).
    pub effects: Vec<Uuid>,

    /// Relationship type.
    pub relationship: String,

    /// Strength.
    pub strength: f32,

    /// Description.
    pub description: String,
}

impl Hyperedge {
    /// Create a new hyperedge.
    pub fn new(
        causes: Vec<Uuid>,
        effects: Vec<Uuid>,
        relationship: impl Into<String>,
        strength: f32,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            causes,
            effects,
            relationship: relationship.into(),
            strength: strength.clamp(0.0, 1.0),
            description: String::new(),
        }
    }
}

/// Causal memory store.
pub struct CausalMemory {
    nodes: HashMap<Uuid, CausalNode>,
    edges: Vec<CausalEdge>,
    hyperedges: Vec<Hyperedge>,
}

impl CausalMemory {
    /// Create a new causal memory.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            hyperedges: Vec::new(),
        }
    }

    /// Add a node.
    pub fn add_node(&mut self, node: CausalNode) -> Uuid {
        let id = node.id;
        self.nodes.insert(id, node);
        id
    }

    /// Add an edge.
    pub fn add_edge(&mut self, edge: CausalEdge) {
        // Check if similar edge exists
        if let Some(existing) = self.edges.iter_mut().find(|e| {
            e.cause == edge.cause && e.effect == edge.effect && e.relationship == edge.relationship
        }) {
            existing.add_evidence(edge.strength);
        } else {
            self.edges.push(edge);
        }
    }

    /// Add a hyperedge.
    pub fn add_hyperedge(&mut self, hyperedge: Hyperedge) {
        self.hyperedges.push(hyperedge);
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: Uuid) -> Option<&CausalNode> {
        self.nodes.get(&id)
    }

    /// Find causes of a given effect.
    pub fn find_causes(&self, effect: Uuid) -> Vec<(&CausalEdge, Option<&CausalNode>)> {
        self.edges
            .iter()
            .filter(|e| e.effect == effect)
            .map(|e| (e, self.nodes.get(&e.cause)))
            .collect()
    }

    /// Find effects of a given cause.
    pub fn find_effects(&self, cause: Uuid) -> Vec<(&CausalEdge, Option<&CausalNode>)> {
        self.edges
            .iter()
            .filter(|e| e.cause == cause)
            .map(|e| (e, self.nodes.get(&e.effect)))
            .collect()
    }

    /// Trace causal chain from cause to all reachable effects.
    pub fn trace_chain(&self, start: Uuid, max_depth: usize) -> Vec<(Uuid, usize, f32)> {
        let mut visited: HashSet<Uuid> = HashSet::new();
        let mut result: Vec<(Uuid, usize, f32)> = Vec::new();
        let mut queue: Vec<(Uuid, usize, f32)> = vec![(start, 0, 1.0)];

        while let Some((current, depth, cumulative_strength)) = queue.pop() {
            if depth > max_depth || visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            if current != start {
                result.push((current, depth, cumulative_strength));
            }

            // Find all effects
            for edge in self.edges.iter().filter(|e| e.cause == current) {
                let new_strength = cumulative_strength * edge.strength;
                if new_strength > 0.1 {
                    // Prune weak chains
                    queue.push((edge.effect, depth + 1, new_strength));
                }
            }
        }

        result
    }

    /// Find all edges of a specific relationship type.
    pub fn find_by_relationship(&self, relationship: &str) -> Vec<&CausalEdge> {
        self.edges
            .iter()
            .filter(|e| e.relationship == relationship)
            .collect()
    }

    /// Get strongest causal relationships.
    pub fn strongest_relationships(&self, limit: usize) -> Vec<&CausalEdge> {
        let mut edges: Vec<_> = self.edges.iter().collect();
        edges.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        edges.into_iter().take(limit).collect()
    }

    /// Node count.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Edge count.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Hyperedge count.
    pub fn hyperedge_count(&self) -> usize {
        self.hyperedges.len()
    }
}

impl Default for CausalMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: Building a causal graph.
    ///
    /// What happens:
    /// 1. Create nodes representing concepts
    /// 2. Add edges representing causal relationships
    /// 3. Query causes and effects
    #[test]
    fn test_causal_graph() {
        let mut memory = CausalMemory::new();

        // Create nodes
        let unwrap_id = memory.add_node(
            CausalNode::new("Using unwrap()", "code_pattern")
                .with_description("Calling .unwrap() on Option/Result"),
        );

        let panic_id = memory.add_node(
            CausalNode::new("Runtime panic", "error")
                .with_description("Program crashes at runtime"),
        );

        let option_handling_id = memory.add_node(
            CausalNode::new("Proper Option handling", "code_pattern")
                .with_description("Using match or if-let"),
        );

        let reliability_id = memory.add_node(
            CausalNode::new("Code reliability", "quality")
                .with_description("Code works correctly in edge cases"),
        );

        // Add causal relationships
        memory.add_edge(CausalEdge::new(unwrap_id, panic_id, "causes", 0.8));
        memory.add_edge(CausalEdge::new(
            option_handling_id,
            reliability_id,
            "improves",
            0.9,
        ));
        memory.add_edge(CausalEdge::new(
            option_handling_id,
            panic_id,
            "prevents",
            0.95,
        ));

        // Query: What causes panics?
        // Both unwrap (causes) and proper handling (prevents) are related to panic
        let panic_causes = memory.find_causes(panic_id);
        assert!(!panic_causes.is_empty());
        // Find the "causes" relationship
        let direct_cause = panic_causes
            .iter()
            .find(|(e, _)| e.relationship == "causes");
        assert!(direct_cause.is_some());
        assert!(direct_cause.unwrap().1.unwrap().label.contains("unwrap"));

        // Query: What does proper handling improve?
        let handling_effects = memory.find_effects(option_handling_id);
        assert_eq!(handling_effects.len(), 2);
    }

    /// Test: Causal chain tracing.
    ///
    /// What happens:
    /// 1. Build a chain: A → B → C → D
    /// 2. Trace from A
    /// 3. Get all reachable nodes with depths and cumulative strength
    #[test]
    fn test_causal_chain() {
        let mut memory = CausalMemory::new();

        // A → B → C → D chain
        let a = memory.add_node(CausalNode::new("A", "concept"));
        let b = memory.add_node(CausalNode::new("B", "concept"));
        let c = memory.add_node(CausalNode::new("C", "concept"));
        let d = memory.add_node(CausalNode::new("D", "concept"));

        memory.add_edge(CausalEdge::new(a, b, "causes", 0.9));
        memory.add_edge(CausalEdge::new(b, c, "causes", 0.8));
        memory.add_edge(CausalEdge::new(c, d, "causes", 0.7));

        // Trace from A
        let chain = memory.trace_chain(a, 10);

        assert_eq!(chain.len(), 3); // B, C, D

        // Check depths
        let b_entry = chain.iter().find(|(id, _, _)| *id == b).unwrap();
        assert_eq!(b_entry.1, 1); // Depth 1

        let d_entry = chain.iter().find(|(id, _, _)| *id == d).unwrap();
        assert_eq!(d_entry.1, 3); // Depth 3

        // Check cumulative strength decays
        assert!(d_entry.2 < b_entry.2); // D has lower cumulative strength
    }

    /// Test: Evidence accumulation.
    ///
    /// What happens:
    /// 1. Observe a causal relationship multiple times
    /// 2. Strength gets updated with Bayesian averaging
    /// 3. More evidence = more reliable estimate
    #[test]
    fn test_evidence_accumulation() {
        let mut memory = CausalMemory::new();

        let cause = Uuid::new_v4();
        let effect = Uuid::new_v4();

        // First observation: strength 0.8
        memory.add_edge(CausalEdge::new(cause, effect, "causes", 0.8));

        // Second observation: strength 0.9
        memory.add_edge(CausalEdge::new(cause, effect, "causes", 0.9));

        // Third observation: strength 0.85
        memory.add_edge(CausalEdge::new(cause, effect, "causes", 0.85));

        // Should be one edge with accumulated evidence
        assert_eq!(memory.edge_count(), 1);

        let edge = &memory.edges[0];
        assert_eq!(edge.evidence_count, 3);

        // Strength should be average-ish of observations
        assert!(edge.strength > 0.8 && edge.strength < 0.9);
    }

    /// Test: Hyperedge for complex causation.
    ///
    /// What happens:
    /// 1. Create a hyperedge: (A AND B) → C
    /// 2. This represents that both A and B are needed to cause C
    #[test]
    fn test_hyperedge() {
        let mut memory = CausalMemory::new();

        let fuel = memory.add_node(CausalNode::new("Fuel", "resource"));
        let spark = memory.add_node(CausalNode::new("Spark", "event"));
        let oxygen = memory.add_node(CausalNode::new("Oxygen", "resource"));
        let fire = memory.add_node(CausalNode::new("Fire", "outcome"));

        // Fire requires fuel AND spark AND oxygen
        memory.add_hyperedge(Hyperedge::new(
            vec![fuel, spark, oxygen],
            vec![fire],
            "causes",
            0.99,
        ));

        assert_eq!(memory.hyperedge_count(), 1);
        assert_eq!(memory.hyperedges[0].causes.len(), 3);
    }
}
