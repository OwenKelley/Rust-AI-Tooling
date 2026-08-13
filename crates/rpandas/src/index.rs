//! Index types — thin slice of `pandas.Index` / `RangeIndex` / `DatetimeIndex`.

use crate::datetime::DatetimeIndex;

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

/// Frame/Series index: range or datetime (`df.index`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Index {
    Range(RangeIndex),
    Datetime(DatetimeIndex),
}

impl Index {
    pub fn range(n: usize) -> Self {
        Self::Range(RangeIndex::new(n))
    }

    pub fn datetime(dt: DatetimeIndex) -> Self {
        Self::Datetime(dt)
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Range(r) => r.len(),
            Self::Datetime(d) => d.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reindex(n: usize) -> Self {
        Self::range(n)
    }

    pub fn as_datetime(&self) -> Option<&DatetimeIndex> {
        match self {
            Self::Datetime(d) => Some(d),
            Self::Range(_) => None,
        }
    }

    pub fn as_range(&self) -> Option<&RangeIndex> {
        match self {
            Self::Range(r) => Some(r),
            Self::Datetime(_) => None,
        }
    }

    /// Contiguous row slice; preserves datetime labels when present.
    pub fn slice_rows(&self, start: usize, end: usize) -> Self {
        match self {
            Self::Range(r) => Self::Range(r.slice(start, end)),
            Self::Datetime(d) => {
                let n = d.len();
                let s = start.min(n);
                let e = end.min(n).max(s);
                Self::Datetime(DatetimeIndex::from_ns(d.values()[s..e].to_vec()))
            }
        }
    }

    /// Fancy take; preserves datetime labels when present.
    pub fn take_rows(&self, indices: &[usize]) -> Self {
        match self {
            Self::Range(_) => Self::reindex(indices.len()),
            Self::Datetime(d) => {
                let vals: Vec<i64> = indices
                    .iter()
                    .map(|&i| {
                        assert!(i < d.len(), "index take out of bounds");
                        d.values()[i]
                    })
                    .collect();
                Self::Datetime(DatetimeIndex::from_ns(vals))
            }
        }
    }
}

impl From<RangeIndex> for Index {
    fn from(r: RangeIndex) -> Self {
        Self::Range(r)
    }
}

impl From<DatetimeIndex> for Index {
    fn from(d: DatetimeIndex) -> Self {
        Self::Datetime(d)
    }
}
