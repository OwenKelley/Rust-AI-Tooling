//! `rpolars` — Polars-shaped DataFrame API for Rust (`std` only, on `rarrow`).

pub mod expr;
pub mod frame;
pub mod groupby;
pub mod io;
pub mod join;
pub mod lazy;
pub mod series;

pub use expr::{col, lit_f64, lit_i64, Expr};
pub use frame::DataFrame;
pub use groupby::{Agg, GroupBy};
pub use io::{read_csv, read_csv_str, write_csv, write_csv_string};
pub use join::JoinHow;
pub use lazy::{LazyFrame, LazyOp};
pub use series::Series;

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_df() -> DataFrame {
        DataFrame::new(vec![
            Series::from_f64("a", vec![1.0, 2.0, 3.0, 4.0]),
            Series::from_i64("b", vec![Some(10), Some(20), None, Some(40)]),
            Series::from_utf8(
                "k",
                vec![
                    Some("x".into()),
                    Some("y".into()),
                    Some("x".into()),
                    Some("y".into()),
                ],
            ),
        ])
    }

    #[test]
    fn select_filter_groupby() {
        let df = sample_df();
        let s = df.select(&["a", "k"]);
        assert_eq!(s.width(), 2);
        let f = df.filter(&col("a").gt(lit_f64(2.0)));
        assert_eq!(f.height(), 2);
        let g = df.groupby(&["k"]).agg(&[("a", Agg::Sum), ("a", Agg::Count)]);
        assert_eq!(g.height(), 2);
    }

    #[test]
    fn join_sort_csv_lazy() {
        let left = sample_df();
        let right = DataFrame::new(vec![
            Series::from_utf8("k", vec![Some("x".into()), Some("y".into())]),
            Series::from_f64("v", vec![100.0, 200.0]),
        ]);
        let j = left.join(&right, &["k"], JoinHow::Inner);
        assert_eq!(j.height(), 4);
        let sorted = left.sort(&["a"], true);
        assert_eq!(sorted.height(), 4);
        let csv = write_csv_string(&left);
        let back = read_csv_str(&csv);
        assert_eq!(back.height(), 4);
        let lazy = left
            .lazy()
            .filter(col("a").gt(lit_f64(1.0)))
            .select(&["a", "k"])
            .collect();
        assert_eq!(lazy.height(), 3);
        assert_eq!(lazy.width(), 2);
    }
}
