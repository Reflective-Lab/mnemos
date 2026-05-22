//! Math utilities shared across mnemos modules.

/// Compute cosine similarity between two vectors.
///
/// Returns a value in `[-1.0, 1.0]` (typically `[0.0, 1.0]` for non-negative
/// embeddings).  Returns `0.0` when either vector has zero norm or the vectors
/// differ in length, treating those degenerate cases as maximally dissimilar.
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: cosine_similarity returns 1.0 for identical unit vectors.
    ///
    /// Protects the shared implementation used by knowledge_base, batch, embedding,
    /// and meta modules.
    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    /// Test: cosine_similarity returns 0.0 for orthogonal vectors.
    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }

    /// Test: cosine_similarity returns 0.0 for zero-norm vectors (degenerate case).
    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0f32, 0.0, 0.0];
        let z = vec![0.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&a, &z) - 0.0).abs() < f32::EPSILON);
        assert!((cosine_similarity(&z, &a) - 0.0).abs() < f32::EPSILON);
    }

    /// Test: cosine_similarity returns 0.0 for mismatched lengths.
    #[test]
    fn test_cosine_similarity_length_mismatch() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < f32::EPSILON);
    }

    /// Test: cosine_similarity returns 0.0 for empty slices.
    #[test]
    fn test_cosine_similarity_empty() {
        assert!((cosine_similarity(&[], &[]) - 0.0).abs() < f32::EPSILON);
    }
}
