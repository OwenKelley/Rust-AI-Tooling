//! CSV I/O — Polars-shaped (`read_csv` / `write_csv`), local `std` only.

use std::fs;
use std::path::Path;

use rarrow::{Array, Float64Array, Int64Array, StringArray};

use crate::frame::DataFrame;
use crate::series::Series;

pub fn write_csv_string(df: &DataFrame) -> String {
    let names = df.get_column_names();
    let mut out = String::new();
    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&escape_csv(name));
    }
    out.push('\n');
    for r in 0..df.height() {
        for (c, s) in df.columns().iter().enumerate() {
            if c > 0 {
                out.push(',');
            }
            out.push_str(&escape_csv(&csv_cell(&s.data, r)));
        }
        out.push('\n');
    }
    out
}

pub fn write_csv(df: &DataFrame, path: impl AsRef<Path>) -> std::io::Result<()> {
    fs::write(path, write_csv_string(df))
}

pub fn read_csv_str(text: &str) -> DataFrame {
    let rows = parse_csv(text);
    assert!(!rows.is_empty(), "read_csv: empty");
    let header = &rows[0];
    let body = &rows[1..];
    let ncols = header.len();
    let mut raw: Vec<Vec<String>> = vec![Vec::with_capacity(body.len()); ncols];
    for row in body {
        assert_eq!(row.len(), ncols, "read_csv: ragged row");
        for (j, cell) in row.iter().enumerate() {
            raw[j].push(cell.clone());
        }
    }
    let cols: Vec<Series> = header
        .iter()
        .enumerate()
        .map(|(j, name)| Series::new(name.clone(), infer_column(&raw[j])))
        .collect();
    DataFrame::new(cols)
}

pub fn read_csv(path: impl AsRef<Path>) -> std::io::Result<DataFrame> {
    Ok(read_csv_str(&fs::read_to_string(path)?))
}

fn csv_cell(data: &Array, r: usize) -> String {
    match data {
        Array::Float64(a) => {
            if a.nulls[r] {
                String::new()
            } else {
                let v = a.values[r];
                if v.fract() == 0.0 && v.abs() < 1e15 {
                    format!("{}", v as i64)
                } else {
                    format!("{v}")
                }
            }
        }
        Array::Int64(a) => {
            if a.nulls[r] {
                String::new()
            } else {
                format!("{}", a.values[r])
            }
        }
        Array::Boolean(a) => {
            if a.nulls[r] {
                String::new()
            } else {
                format!("{}", a.values[r])
            }
        }
        Array::Utf8(a) => a.values[r].clone().unwrap_or_default(),
        Array::TimestampNs(a) => {
            if a.nulls[r] {
                String::new()
            } else {
                format!("{}", a.values[r])
            }
        }
        Array::ListFloat64(_) => String::from("[]"),
        Array::DictionaryUtf8(a) => {
            if a.nulls[r] {
                String::new()
            } else {
                a.dictionary[a.indices[r] as usize].clone()
            }
        }
    }
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
    let mut chars = text.chars().peekable();
    let mut in_quotes = false;
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
                    if !(row.len() == 1 && row[0].is_empty() && rows.is_empty()) {
                        rows.push(std::mem::take(&mut row));
                    } else {
                        row.clear();
                    }
                }
                '\r' => {}
                _ => field.push(ch),
            }
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

fn infer_column(cells: &[String]) -> Array {
    if cells.iter().all(|c| c.is_empty()) {
        return Array::Float64(Float64Array {
            values: vec![0.0; cells.len()],
            nulls: vec![true; cells.len()],
        });
    }
    let mut all_int = true;
    let mut all_float = true;
    let mut ints = Vec::with_capacity(cells.len());
    let mut floats = Vec::with_capacity(cells.len());
    let mut inulls = Vec::with_capacity(cells.len());
    let mut fnulls = Vec::with_capacity(cells.len());
    for c in cells {
        if c.is_empty() {
            ints.push(0);
            inulls.push(true);
            floats.push(0.0);
            fnulls.push(true);
            continue;
        }
        if let Ok(v) = c.parse::<i64>() {
            ints.push(v);
            inulls.push(false);
            floats.push(v as f64);
            fnulls.push(false);
        } else if let Ok(v) = c.parse::<f64>() {
            all_int = false;
            ints.push(0);
            inulls.push(true);
            floats.push(v);
            fnulls.push(false);
        } else {
            all_int = false;
            all_float = false;
            break;
        }
    }
    if all_int {
        Array::Int64(Int64Array {
            values: ints,
            nulls: inulls,
        })
    } else if all_float {
        Array::Float64(Float64Array {
            values: floats,
            nulls: fnulls,
        })
    } else {
        Array::Utf8(StringArray {
            values: cells
                .iter()
                .map(|c| if c.is_empty() { None } else { Some(c.clone()) })
                .collect(),
        })
    }
}
