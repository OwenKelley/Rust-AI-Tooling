//! Expression tree v1 — column comparisons + and/or.

use rarrow::{Array, BooleanArray};

use crate::frame::DataFrame;

#[derive(Debug, Clone)]
pub enum Expr {
    Col(String),
    LitF64(f64),
    LitI64(i64),
    Gt(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Eq(Box<Expr>, Box<Expr>),
    Neq(Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
}

pub fn col(name: impl Into<String>) -> Expr {
    Expr::Col(name.into())
}

pub fn lit_f64(v: f64) -> Expr {
    Expr::LitF64(v)
}

pub fn lit_i64(v: i64) -> Expr {
    Expr::LitI64(v)
}

impl Expr {
    pub fn gt(self, other: Expr) -> Expr {
        Expr::Gt(Box::new(self), Box::new(other))
    }
    pub fn ge(self, other: Expr) -> Expr {
        Expr::Ge(Box::new(self), Box::new(other))
    }
    pub fn lt(self, other: Expr) -> Expr {
        Expr::Lt(Box::new(self), Box::new(other))
    }
    pub fn le(self, other: Expr) -> Expr {
        Expr::Le(Box::new(self), Box::new(other))
    }
    pub fn eq(self, other: Expr) -> Expr {
        Expr::Eq(Box::new(self), Box::new(other))
    }
    pub fn neq(self, other: Expr) -> Expr {
        Expr::Neq(Box::new(self), Box::new(other))
    }
    pub fn and(self, other: Expr) -> Expr {
        Expr::And(Box::new(self), Box::new(other))
    }
    pub fn or(self, other: Expr) -> Expr {
        Expr::Or(Box::new(self), Box::new(other))
    }

    /// Evaluate to a boolean mask (`true` = keep row). Null comparisons → false.
    pub fn eval_bool(&self, df: &DataFrame) -> Vec<bool> {
        match self {
            Expr::And(a, b) => a
                .eval_bool(df)
                .into_iter()
                .zip(b.eval_bool(df))
                .map(|(x, y)| x && y)
                .collect(),
            Expr::Or(a, b) => a
                .eval_bool(df)
                .into_iter()
                .zip(b.eval_bool(df))
                .map(|(x, y)| x || y)
                .collect(),
            Expr::Gt(a, b) => cmp_mask(df, a, b, CmpOp::Gt),
            Expr::Ge(a, b) => cmp_mask(df, a, b, CmpOp::Ge),
            Expr::Lt(a, b) => cmp_mask(df, a, b, CmpOp::Lt),
            Expr::Le(a, b) => cmp_mask(df, a, b, CmpOp::Le),
            Expr::Eq(a, b) => cmp_mask(df, a, b, CmpOp::Eq),
            Expr::Neq(a, b) => cmp_mask(df, a, b, CmpOp::Neq),
            Expr::Col(name) => match &df.column(name).data {
                Array::Boolean(a) => a
                    .values
                    .iter()
                    .zip(a.nulls.iter())
                    .map(|(&v, &n)| !n && v)
                    .collect(),
                other => panic!("filter predicate column must be bool, got {other:?}"),
            },
            Expr::LitF64(_) | Expr::LitI64(_) => {
                panic!("literal alone is not a boolean predicate")
            }
        }
    }
}

#[derive(Clone, Copy)]
enum CmpOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Neq,
}

enum Scalar {
    F64(Vec<Option<f64>>),
    I64(Vec<Option<i64>>),
}

fn eval_numeric(df: &DataFrame, expr: &Expr) -> Scalar {
    let n = df.height();
    match expr {
        Expr::Col(name) => match &df.column(name).data {
            Array::Float64(a) => Scalar::F64(
                a.values
                    .iter()
                    .zip(a.nulls.iter())
                    .map(|(&v, &n)| if n { None } else { Some(v) })
                    .collect(),
            ),
            Array::Int64(a) => Scalar::I64(
                a.values
                    .iter()
                    .zip(a.nulls.iter())
                    .map(|(&v, &n)| if n { None } else { Some(v) })
                    .collect(),
            ),
            Array::Boolean(a) => Scalar::I64(
                a.values
                    .iter()
                    .zip(a.nulls.iter())
                    .map(|(&v, &n)| if n { None } else { Some(i64::from(v)) })
                    .collect(),
            ),
            Array::Utf8(_) => panic!("utf8 not supported in numeric comparisons"),
            Array::TimestampNs(a) => Scalar::I64(
                a.values
                    .iter()
                    .zip(a.nulls.iter())
                    .map(|(&v, &n)| if n { None } else { Some(v) })
                    .collect(),
            ),
            Array::ListFloat64(_) => panic!("list not supported in numeric comparisons"),
            Array::DictionaryUtf8(_) => panic!("dictionary not supported in numeric comparisons"),
        },
        Expr::LitF64(v) => Scalar::F64(vec![Some(*v); n]),
        Expr::LitI64(v) => Scalar::I64(vec![Some(*v); n]),
        _ => panic!("nested bool exprs not valid as numeric operands"),
    }
}

fn cmp_mask(df: &DataFrame, left: &Expr, right: &Expr, op: CmpOp) -> Vec<bool> {
    let l = eval_numeric(df, left);
    let r = eval_numeric(df, right);
    match (l, r) {
        (Scalar::F64(a), Scalar::F64(b)) => cmp_opt_f64(&a, &b, op),
        (Scalar::I64(a), Scalar::I64(b)) => cmp_opt_i64(&a, &b, op),
        (Scalar::F64(a), Scalar::I64(b)) => {
            let b: Vec<Option<f64>> = b.iter().map(|v| v.map(|x| x as f64)).collect();
            cmp_opt_f64(&a, &b, op)
        }
        (Scalar::I64(a), Scalar::F64(b)) => {
            let a: Vec<Option<f64>> = a.iter().map(|v| v.map(|x| x as f64)).collect();
            cmp_opt_f64(&a, &b, op)
        }
    }
}

fn cmp_opt_f64(a: &[Option<f64>], b: &[Option<f64>], op: CmpOp) -> Vec<bool> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| match (x, y) {
            (Some(x), Some(y)) => match op {
                CmpOp::Gt => x > y,
                CmpOp::Ge => x >= y,
                CmpOp::Lt => x < y,
                CmpOp::Le => x <= y,
                CmpOp::Eq => x.to_bits() == y.to_bits(),
                CmpOp::Neq => x.to_bits() != y.to_bits(),
            },
            _ => false,
        })
        .collect()
}

fn cmp_opt_i64(a: &[Option<i64>], b: &[Option<i64>], op: CmpOp) -> Vec<bool> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| match (x, y) {
            (Some(x), Some(y)) => match op {
                CmpOp::Gt => x > y,
                CmpOp::Ge => x >= y,
                CmpOp::Lt => x < y,
                CmpOp::Le => x <= y,
                CmpOp::Eq => x == y,
                CmpOp::Neq => x != y,
            },
            _ => false,
        })
        .collect()
}

/// Build a bool Series from a mask (helper for tests).
pub fn mask_to_series(name: &str, mask: &[bool]) -> crate::series::Series {
    crate::series::Series::new(
        name,
        Array::Boolean(BooleanArray {
            values: mask.to_vec(),
            nulls: vec![false; mask.len()],
        }),
    )
}
