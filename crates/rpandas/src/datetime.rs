//! Datetime index helpers — pandas `DatetimeIndex` / `date_range` (naive ns).
//!
//! Timestamps are UTC-naive nanoseconds since the Unix epoch (pandas
//! `datetime64[ns]` style). No timezones in v1.

/// Fixed frequency offsets (pandas aliases `'h'` / `'D'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freq {
    /// 1 hour
    H,
    /// 1 calendar day (24h)
    D,
}

impl Freq {
    pub fn as_ns(self) -> i64 {
        match self {
            Freq::H => 3_600_000_000_000,
            Freq::D => 86_400_000_000_000,
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "h" | "H" | "hour" | "hours" => Freq::H,
            "D" | "d" | "day" | "days" => Freq::D,
            other => panic!("unsupported freq '{other}' (v1 supports h/D only)"),
        }
    }
}

/// Ordered datetime labels as epoch nanoseconds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatetimeIndex {
    values: Vec<i64>,
}

impl DatetimeIndex {
    pub fn from_ns(values: Vec<i64>) -> Self {
        Self { values }
    }

    pub fn values(&self) -> &[i64] {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// `pd.date_range(start, periods=…, freq=…)` with `start` as epoch ns.
pub fn date_range(start_ns: i64, periods: usize, freq: Freq) -> DatetimeIndex {
    let step = freq.as_ns();
    let mut values = Vec::with_capacity(periods);
    let mut t = start_ns;
    for _ in 0..periods {
        values.push(t);
        t = t.checked_add(step).expect("date_range overflow");
    }
    DatetimeIndex { values }
}

/// Floor timestamp into a left-closed bin edge: `floor(t / period) * period`.
pub fn floor_bin(t: i64, period_ns: i64) -> i64 {
    assert!(period_ns > 0);
    if t >= 0 {
        (t / period_ns) * period_ns
    } else {
        let q = t / period_ns;
        if t % period_ns == 0 {
            q * period_ns
        } else {
            (q - 1) * period_ns
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_range_hourly() {
        let idx = date_range(0, 3, Freq::H);
        assert_eq!(idx.values(), &[0, 3_600_000_000_000, 7_200_000_000_000]);
    }

    #[test]
    fn floor_bin_positive() {
        let p = Freq::H.as_ns();
        assert_eq!(floor_bin(0, p), 0);
        assert_eq!(floor_bin(p - 1, p), 0);
        assert_eq!(floor_bin(p, p), p);
    }
}
