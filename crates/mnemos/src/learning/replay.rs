//! Experience replay buffer for stable learning.

use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::VecDeque;

/// Experience replay buffer using reservoir sampling.
///
/// This ensures uniform coverage of past experiences for stable learning,
/// preventing catastrophic forgetting as described in the ruvector-gnn crate.
pub struct ReplayBuffer<T> {
    /// Maximum buffer capacity.
    capacity: usize,

    /// Stored experiences.
    buffer: VecDeque<T>,

    /// Total experiences seen (for reservoir sampling).
    total_seen: u64,
}

impl<T: Clone> ReplayBuffer<T> {
    /// Create a new replay buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
            total_seen: 0,
        }
    }

    /// Add an experience to the buffer using reservoir sampling.
    pub fn add(&mut self, experience: T) {
        self.total_seen += 1;

        if self.buffer.len() < self.capacity {
            // Buffer not full: just add
            self.buffer.push_back(experience);
        } else {
            // Reservoir sampling: replace with probability capacity/total_seen
            let mut rng = rand::thread_rng();
            let replace_prob = self.capacity as f64 / self.total_seen as f64;

            if rng.r#gen::<f64>() < replace_prob {
                let idx = rng.gen_range(0..self.capacity);
                self.buffer[idx] = experience;
            }
        }
    }

    /// Sample n experiences uniformly at random.
    pub fn sample(&self, n: usize) -> Vec<T> {
        if self.buffer.is_empty() {
            return Vec::new();
        }

        let n = n.min(self.buffer.len());
        let mut rng = rand::thread_rng();

        let indices: Vec<usize> = {
            let mut all_indices: Vec<usize> = (0..self.buffer.len()).collect();
            all_indices.shuffle(&mut rng);
            all_indices.into_iter().take(n).collect()
        };

        indices
            .into_iter()
            .filter_map(|i| self.buffer.get(i).cloned())
            .collect()
    }

    /// Get the current buffer size.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get total experiences seen.
    pub fn total_seen(&self) -> u64 {
        self.total_seen
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.total_seen = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_buffer_add() {
        let mut buffer = ReplayBuffer::new(100);

        for i in 0..50 {
            buffer.add(i);
        }

        assert_eq!(buffer.len(), 50);
        assert_eq!(buffer.total_seen(), 50);
    }

    #[test]
    fn test_replay_buffer_reservoir() {
        let mut buffer = ReplayBuffer::new(10);

        for i in 0..1000 {
            buffer.add(i);
        }

        assert_eq!(buffer.len(), 10);
        assert_eq!(buffer.total_seen(), 1000);
    }

    #[test]
    fn test_replay_buffer_sample() {
        let mut buffer = ReplayBuffer::new(100);

        for i in 0..100 {
            buffer.add(i);
        }

        let samples = buffer.sample(10);
        assert_eq!(samples.len(), 10);
    }
}
