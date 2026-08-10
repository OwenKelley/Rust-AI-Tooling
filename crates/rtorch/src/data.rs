//! `torch.utils.data` — TensorDataset + DataLoader + samplers (CPU, local/`std`).

use crate::ops::{select, stack};
use crate::tensor::Tensor;

/// `torch.utils.data.TensorDataset(features, labels)`.
pub struct TensorDataset {
    pub features: Tensor,
    pub labels: Tensor,
}

impl TensorDataset {
    pub fn new(features: Tensor, labels: Tensor) -> Self {
        assert_eq!(
            features.shape()[0],
            labels.shape()[0],
            "TensorDataset: leading dims must match"
        );
        Self { features, labels }
    }

    pub fn len(&self) -> usize {
        self.features.shape()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn fisher_yates(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state = seed;
    for i in (1..n).rev() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let j = ((state >> 8) as usize) % (i + 1);
        idx.swap(i, j);
    }
    idx
}

/// `torch.utils.data.SequentialSampler` — indices `0..len`.
pub struct SequentialSampler {
    pub len: usize,
}

impl SequentialSampler {
    pub fn new(len: usize) -> Self {
        Self { len }
    }

    pub fn indices(&self) -> Vec<usize> {
        (0..self.len).collect()
    }
}

/// `torch.utils.data.RandomSampler` — LCG Fisher–Yates permutation (replacement=False).
pub struct RandomSampler {
    pub len: usize,
    pub seed: u64,
}

impl RandomSampler {
    pub fn new(len: usize, seed: u64) -> Self {
        Self { len, seed }
    }

    pub fn indices(&self) -> Vec<usize> {
        fisher_yates(self.len, self.seed)
    }
}

/// `torch.utils.data.default_collate` for a list of `(features, labels)` samples —
/// stacks on dim 0.
pub fn default_collate(batch: &[(Tensor, Tensor)]) -> (Tensor, Tensor) {
    assert!(!batch.is_empty(), "default_collate: empty batch");
    let xs: Vec<&Tensor> = batch.iter().map(|(x, _)| x).collect();
    let ys: Vec<&Tensor> = batch.iter().map(|(_, y)| y).collect();
    (stack(&xs, 0), stack(&ys, 0))
}

/// `torch.utils.data.DataLoader` — batches over sampler indices.
pub struct DataLoader {
    features: Tensor,
    labels: Tensor,
    batch_size: usize,
    indices: Vec<usize>,
    pos: usize,
}

impl DataLoader {
    pub fn new(dataset: &TensorDataset, batch_size: usize, shuffle: bool, seed: u64) -> Self {
        let indices = if shuffle {
            RandomSampler::new(dataset.len(), seed).indices()
        } else {
            SequentialSampler::new(dataset.len()).indices()
        };
        Self::from_indices(dataset, batch_size, indices)
    }

    pub fn from_sequential(dataset: &TensorDataset, batch_size: usize) -> Self {
        Self::from_indices(
            dataset,
            batch_size,
            SequentialSampler::new(dataset.len()).indices(),
        )
    }

    pub fn from_random(dataset: &TensorDataset, batch_size: usize, seed: u64) -> Self {
        Self::from_indices(
            dataset,
            batch_size,
            RandomSampler::new(dataset.len(), seed).indices(),
        )
    }

    pub fn from_indices(dataset: &TensorDataset, batch_size: usize, indices: Vec<usize>) -> Self {
        assert!(batch_size > 0);
        Self {
            features: dataset.features.clone(),
            labels: dataset.labels.clone(),
            batch_size,
            indices,
            pos: 0,
        }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Next `(features_batch, labels_batch)`, or `None` when exhausted.
    pub fn next_batch(&mut self) -> Option<(Tensor, Tensor)> {
        if self.pos >= self.indices.len() {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.indices.len());
        let batch_idx = &self.indices[self.pos..end];
        self.pos = end;
        // Gather samples then `default_collate` (stack on dim 0).
        let mut samples = Vec::with_capacity(batch_idx.len());
        for &i in batch_idx {
            let x = select(&self.features, 0, i);
            let y = select(&self.labels, 0, i);
            samples.push((x, y));
        }
        Some(default_collate(&samples))
    }

    /// Sum of checksums over one epoch (features + labels per batch).
    pub fn epoch_checksum(&mut self) -> f64 {
        self.reset();
        let mut acc = 0.0f64;
        while let Some((x, y)) = self.next_batch() {
            acc += x.checksum() + y.checksum();
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::seeded_uniform;

    #[test]
    fn sequential_vs_random() {
        let n = 16;
        let ds = TensorDataset::new(
            seeded_uniform(&[n, 3], 1, -1.0, 1.0),
            seeded_uniform(&[n, 1], 2, -1.0, 1.0),
        );
        let seq = SequentialSampler::new(n).indices();
        assert_eq!(seq, (0..n).collect::<Vec<_>>());
        let rnd = RandomSampler::new(n, 99).indices();
        assert_eq!(rnd.len(), n);
        assert_ne!(rnd, seq);
        let mut a = DataLoader::from_sequential(&ds, 4);
        let mut b = DataLoader::from_random(&ds, 4, 99);
        let (xa, _) = a.next_batch().unwrap();
        let (xb, _) = b.next_batch().unwrap();
        assert!((xa.checksum() - xb.checksum()).abs() > 1e-6);
    }
}
