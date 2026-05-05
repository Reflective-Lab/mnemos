//! GNN layer implementation inspired by ruvector-gnn.

use rand::Rng;

/// A simplified GNN layer for embedding transformation.
///
/// This implements a message-passing mechanism where node embeddings
/// are updated based on their neighbors, similar to the ruvector GNN layer.
pub struct GnnLayer {
    /// Input dimensions.
    input_dim: usize,

    /// Hidden dimensions (reserved for future multi-layer expansion).
    #[allow(dead_code)]
    hidden_dim: usize,

    /// Number of attention heads.
    num_heads: usize,

    /// Linear transformation weights.
    weights: Vec<f32>,

    /// Attention weights (reserved for future explicit attention-head reads).
    #[allow(dead_code)]
    attention_weights: Vec<f32>,
}

impl GnnLayer {
    /// Create a new GNN layer.
    pub fn new(input_dim: usize, hidden_dim: usize, num_heads: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let scale = (2.0 / (input_dim + hidden_dim) as f32).sqrt();

        let weights: Vec<f32> = (0..input_dim * hidden_dim)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();

        let attention_weights: Vec<f32> = (0..num_heads * input_dim)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();

        Self {
            input_dim,
            hidden_dim,
            num_heads,
            weights,
            attention_weights,
        }
    }

    /// Forward pass through the GNN layer.
    pub fn forward(
        &self,
        node_embedding: &[f32],
        neighbor_embeddings: &[Vec<f32>],
        edge_weights: &[f32],
    ) -> Vec<f32> {
        if neighbor_embeddings.is_empty() {
            // No neighbors: just apply linear transformation
            return self.linear_transform(node_embedding);
        }

        // Compute attention scores for each neighbor
        let attention_scores = self.compute_attention(node_embedding, neighbor_embeddings);

        // Combine attention with edge weights
        let combined_weights: Vec<f32> = attention_scores
            .iter()
            .zip(edge_weights.iter())
            .map(|(a, e)| a * e)
            .collect();

        // Normalize weights
        let weight_sum: f32 = combined_weights.iter().sum();
        let normalized_weights: Vec<f32> = if weight_sum > 0.0 {
            combined_weights.iter().map(|w| w / weight_sum).collect()
        } else {
            vec![1.0 / neighbor_embeddings.len() as f32; neighbor_embeddings.len()]
        };

        // Aggregate neighbor messages
        let mut aggregated = vec![0.0f32; self.input_dim];
        for (neighbor, &weight) in neighbor_embeddings.iter().zip(normalized_weights.iter()) {
            for (i, &val) in neighbor.iter().enumerate() {
                if i < self.input_dim {
                    aggregated[i] += val * weight;
                }
            }
        }

        // Combine with node embedding (skip connection)
        let combined: Vec<f32> = node_embedding
            .iter()
            .zip(aggregated.iter())
            .map(|(n, a)| 0.5 * n + 0.5 * a)
            .collect();

        // Apply linear transformation
        let transformed = self.linear_transform(&combined);

        // Apply ReLU activation
        transformed.into_iter().map(|x| x.max(0.0)).collect()
    }

    /// Compute attention scores for neighbors.
    fn compute_attention(&self, query: &[f32], keys: &[Vec<f32>]) -> Vec<f32> {
        let head_dim = self.input_dim / self.num_heads;

        let scores: Vec<f32> = keys
            .iter()
            .map(|key| {
                let mut score = 0.0f32;
                for h in 0..self.num_heads {
                    let start = h * head_dim;
                    let end = (start + head_dim).min(query.len()).min(key.len());

                    let dot: f32 = query[start..end]
                        .iter()
                        .zip(key[start..end].iter())
                        .map(|(q, k)| q * k)
                        .sum();

                    score += dot / (head_dim as f32).sqrt();
                }
                score / self.num_heads as f32
            })
            .collect();

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();

        exp_scores.iter().map(|e| e / sum_exp).collect()
    }

    /// Apply linear transformation.
    fn linear_transform(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; self.input_dim];

        for i in 0..self.input_dim {
            for j in 0..input.len().min(self.input_dim) {
                let weight_idx = i * self.input_dim + j;
                if weight_idx < self.weights.len() {
                    output[i] += input[j] * self.weights[weight_idx];
                }
            }
        }

        output
    }

    /// Update weights based on feedback.
    pub fn update(&mut self, query: &[f32], target_score: f32, learning_rate: f32) {
        // Simplified gradient update
        for (i, weight) in self.weights.iter_mut().enumerate() {
            let query_idx = i % query.len().max(1);
            let gradient = query.get(query_idx).unwrap_or(&0.0) * (target_score - 1.0);
            *weight += learning_rate * gradient * 0.01;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gnn_layer_forward() {
        let layer = GnnLayer::new(64, 128, 4);

        let node = vec![0.5; 64];
        let neighbors = vec![vec![0.3; 64], vec![0.7; 64]];
        let edge_weights = vec![0.8, 0.6];

        let output = layer.forward(&node, &neighbors, &edge_weights);

        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_gnn_layer_no_neighbors() {
        let layer = GnnLayer::new(32, 64, 2);

        let node = vec![0.5; 32];
        let output = layer.forward(&node, &[], &[]);

        assert_eq!(output.len(), 32);
    }
}
