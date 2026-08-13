//! Lazy frame v1 — plan + collect for filter / select / groupby.

use crate::expr::Expr;
use crate::frame::DataFrame;
use crate::groupby::Agg;

#[derive(Debug, Clone)]
pub enum LazyOp {
    Select(Vec<String>),
    Filter(Expr),
    GroupBy {
        keys: Vec<String>,
        aggs: Vec<(String, Agg)>,
    },
}

#[derive(Debug, Clone)]
pub struct LazyFrame {
    source: DataFrame,
    plan: Vec<LazyOp>,
}

impl LazyFrame {
    pub fn new(df: DataFrame) -> Self {
        Self {
            source: df,
            plan: Vec::new(),
        }
    }

    pub fn select(mut self, names: &[&str]) -> Self {
        self.plan
            .push(LazyOp::Select(names.iter().map(|s| (*s).to_string()).collect()));
        self
    }

    pub fn filter(mut self, predicate: Expr) -> Self {
        self.plan.push(LazyOp::Filter(predicate));
        self
    }

    pub fn groupby(self, keys: &[&str]) -> LazyGroupBy {
        LazyGroupBy {
            lf: self,
            keys: keys.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    pub fn collect(self) -> DataFrame {
        let mut df = self.source;
        for op in self.plan {
            df = match op {
                LazyOp::Select(names) => {
                    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
                    df.select(&refs)
                }
                LazyOp::Filter(pred) => df.filter(&pred),
                LazyOp::GroupBy { keys, aggs } => {
                    let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
                    let agg_refs: Vec<(&str, Agg)> =
                        aggs.iter().map(|(c, a)| (c.as_str(), *a)).collect();
                    df.groupby(&key_refs).agg(&agg_refs)
                }
            };
        }
        df
    }
}

pub struct LazyGroupBy {
    lf: LazyFrame,
    keys: Vec<String>,
}

impl LazyGroupBy {
    pub fn agg(mut self, aggs: &[(&str, Agg)]) -> LazyFrame {
        self.lf.plan.push(LazyOp::GroupBy {
            keys: self.keys,
            aggs: aggs
                .iter()
                .map(|(c, a)| ((*c).to_string(), *a))
                .collect(),
        });
        self.lf
    }
}
