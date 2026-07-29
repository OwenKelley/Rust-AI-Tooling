//! Simple index types — mirrors a thin slice of `pandas.Index` / `RangeIndex`.

/// Positional row index `0..n` (`pandas.RangeIndex`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeIndex {
    pub start: usize,
    pub stop: usize,
}

impl RangeIndex {
    pub fn new(n: usize) -> Self {
        Self {
            start: 0,
            stop: n,
        }
    }

    pub fn len(&self) -> usize {
        self.stop.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Take a contiguous slice of the index (for `head`/`tail`/`iloc` ranges).
    pub fn slice(&self, start: usize, end: usize) -> Self {
        let n = self.len();
        let s = start.min(n);
        let e = end.min(n).max(s);
        Self {
            start: self.start + s,
            stop: self.start + e,
        }
    }

    /// Rebuild a dense `0..len` index after row filtering/reordering.
    pub fn reindex(n: usize) -> Self {
        Self::new(n)
    }
}
