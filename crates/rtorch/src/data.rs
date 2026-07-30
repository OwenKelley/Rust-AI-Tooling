//! `torch.utils.data` — TensorDataset + DataLoader (CPU, local/`std`).

use crate::ops::index_select;
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

/// `torch.utils.data.DataLoader` — fixed-order or LCG-shuffled batches.
pub struct DataLoader {
    features: Tensor,
    labels: Tensor,
    batch_size: usize,
    indices: Vec<usize>,
    pos: usize,
}

impl DataLoader {
    pub fn new(dataset: &TensorDataset, batch_size: usize, shuffle: bool, seed: u64) -> Self {
        assert!(batch_size > 0);
        let n = dataset.len();
        let indices = if shuffle {
            fisher_yates(n, seed)
        } else {
            (0..n).collect()
        };
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
        let xb = index_select(&self.features, 0, batch_idx);
        let yb = index_select(&self.labels, 0, batch_idx);
        Some((xb, yb))
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
