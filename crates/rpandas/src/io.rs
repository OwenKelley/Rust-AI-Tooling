//! CSV I/O — mirrors `pandas.read_csv` / `DataFrame.to_csv` (local `std` only).

use std::fs;
use std::path::Path;

use rnumpy::NdArray;

use crate::frame::{Column, DataFrame};

/// `df.to_csv(...)` → string (header on, index off — matches common ML export).
pub fn to_csv_string(df: &DataFrame) -> String {
    let names = df.column_names();
    let mut out = String::new();
    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&escape_csv(name));
    }
    out.push('\n');
    let ncols = df.ncols();
    let cols = df.columns_ref();
    for r in 0..df.nrows() {
        for c in 0..ncols {
            if c > 0 {
                out.push(',');
            }
            out.push_str(&escape_csv(&cols[c].1.csv_cell(r)));
        }
        out.push('\n');
    }
    out
}

/// Write CSV to a file path.
pub fn to_csv(df: &DataFrame, path: impl AsRef<Path>) -> std::io::Result<()> {
    fs::write(path, to_csv_string(df))
}

/// `pd.read_csv` from a string (header row required).
///
/// Columns that parse entirely as floats become `Float64` (empty → NaN);
/// otherwise UTF-8. Integers that fit and have no fractional form stay as
/// float when any value has a decimal, else `Int64` if all integral.
pub fn read_csv_str(text: &str) -> DataFrame {
    let rows = parse_csv(text);
    assert!(!rows.is_empty(), "read_csv: empty input");
    let header = &rows[0];
    let ncols = header.len();
    let body = &rows[1..];
    let nrows = body.len();

    let mut raw: Vec<Vec<String>> = vec![Vec::with_capacity(nrows); ncols];
    for row in body {
        assert_eq!(row.len(), ncols, "read_csv: ragged row");
        for (j, cell) in row.iter().enumerate() {
            raw[j].push(cell.clone());
        }
    }

    let mut cols = Vec::with_capacity(ncols);
    for (j, name) in header.iter().enumerate() {
        cols.push((name.clone(), infer_column(&raw[j])));
    }
    DataFrame::from_columns(cols)
}

/// Read CSV from a file path.
pub fn read_csv(path: impl AsRef<Path>) -> std::io::Result<DataFrame> {
    let text = fs::read_to_string(path)?;
    Ok(read_csv_str(&text))
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        let mut out = String::from("\"");
        for ch in s.chars() {
            if ch == '"' {
                out.push('"');
            }
            out.push(ch);
        }
        out.push('"');
        out
    } else {
        s.to_string()
    }
}

fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(ch);
            }
        } else {
            match ch {
                '"' => in_quotes = true,
                ',' => {
                    row.push(std::mem::take(&mut field));
                }
                '\n' => {
                    row.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut row));
                }
                '\r' => {
                    // ignore; handle \r\n via following \n
                }
                _ => field.push(ch),
            }
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    // Drop trailing empty row from final newline
    if let Some(last) = rows.last() {
        if last.len() == 1 && last[0].is_empty() && rows.len() > 1 {
            // only if it looks like blank line — keep cells from proper rows
        }
    }
    if rows.last().map(|r| r.iter().all(|c| c.is_empty())).unwrap_or(false) {
        rows.pop();
    }
    rows
}

fn infer_column(cells: &[String]) -> Column {
    let n = cells.len();

    // Try bool (True/False only; empty → null)
    let mut all_bool = true;
    let mut bools = Vec::with_capacity(n);
    let mut nulls = Vec::with_capacity(n);
    for c in cells {
        let t = c.trim();
        if t.is_empty() {
            bools.push(false);
            nulls.push(true);
        } else if t.eq_ignore_ascii_case("true") {
            bools.push(true);
            nulls.push(false);
        } else if t.eq_ignore_ascii_case("false") {
            bools.push(false);
            nulls.push(false);
        } else {
            all_bool = false;
            break;
        }
    }
    if all_bool {
        return Column::Bool {
            values: bools,
            nulls,
        };
    }

    // Try float / int
    let mut floats = Vec::with_capacity(n);
    let mut all_int = true;
    let mut any_num = false;
    let mut all_num_or_empty = true;
    for c in cells {
        let t = c.trim();
        if t.is_empty() {
            floats.push(f64::NAN);
            continue;
        }
        match t.parse::<f64>() {
            Ok(v) => {
                any_num = true;
                if !(v.fract() == 0.0 && v.abs() <= i64::MAX as f64 && !v.is_nan()) {
                    all_int = false;
                }
                floats.push(v);
            }
            Err(_) => {
                all_num_or_empty = false;
                break;
            }
        }
    }
    if all_num_or_empty && any_num {
        if all_int {
            let mut values = Vec::with_capacity(n);
            let mut nulls = Vec::with_capacity(n);
            for &v in &floats {
                if v.is_nan() {
                    values.push(0);
                    nulls.push(true);
                } else {
                    values.push(v as i64);
                    nulls.push(false);
                }
            }
            return Column::Int64 { values, nulls };
        }
        return Column::Float64(NdArray::from_vec(floats));
    }

    let mut values = Vec::with_capacity(n);
    let mut nulls = Vec::with_capacity(n);
    for c in cells {
        if c.is_empty() {
            values.push(String::new());
            nulls.push(true);
        } else {
            values.push(c.clone());
            nulls.push(false);
        }
    }
    Column::Utf8 { values, nulls }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_roundtrip_numeric() {
        let df = DataFrame::from_columns(vec![
            (
                "a".into(),
                Column::Float64(NdArray::from_vec(vec![1.5, 2.0])),
            ),
            (
                "b".into(),
                Column::Int64 {
                    values: vec![3, 4],
                    nulls: vec![false, false],
                },
            ),
        ]);
        let s = to_csv_string(&df);
        let back = read_csv_str(&s);
        assert_eq!(back.nrows(), 2);
        assert_eq!(back.ncols(), 2);
        assert!((back.float_slice("a")[0] - 1.5).abs() < 1e-12);
    }
}
