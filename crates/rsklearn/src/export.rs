//! ONNX-ish JSON model dump / load for a tiny subset of estimators (`std` only).

use std::fs;
use std::path::Path;

use rnumpy::NdArray;

use crate::linear::{LinearRegression, LogisticRegression};
use crate::preprocessing::StandardScaler;

/// Portable artifact format version.
pub const FORMAT: &str = "rsklearn-onnxish-v1";

#[derive(Debug, Clone, PartialEq)]
pub enum ModelArtifact {
    LinearRegression {
        coef: Vec<f64>,
        intercept: f64,
    },
    LogisticRegression {
        coef: Vec<f64>,
        intercept: f64,
    },
    StandardScaler {
        mean: Vec<f64>,
        scale: Vec<f64>,
    },
}

impl ModelArtifact {
    pub fn from_linear(m: &LinearRegression) -> Self {
        Self::LinearRegression {
            coef: m.coef_.clone(),
            intercept: m.intercept_,
        }
    }

    pub fn from_logistic(m: &LogisticRegression) -> Self {
        Self::LogisticRegression {
            coef: m.coef_.clone(),
            intercept: m.intercept_,
        }
    }

    pub fn from_scaler(m: &StandardScaler) -> Self {
        Self::StandardScaler {
            mean: m.mean_.clone(),
            scale: m.scale_.clone(),
        }
    }

    /// Predict / transform for supported models. `x` is shape `[n, d]`.
    pub fn apply(&self, x: &NdArray) -> Vec<f64> {
        match self {
            Self::LinearRegression { coef, intercept } => {
                let mut lr = LinearRegression::new();
                lr.coef_ = coef.clone();
                lr.intercept_ = *intercept;
                lr.predict(x)
            }
            Self::LogisticRegression { coef, intercept } => {
                let mut lr = LogisticRegression::new();
                lr.coef_ = coef.clone();
                lr.intercept_ = *intercept;
                lr.predict(x).into_iter().map(|y| y as f64).collect()
            }
            Self::StandardScaler { mean, scale } => {
                let mut sc = StandardScaler::new();
                sc.mean_ = mean.clone();
                sc.scale_ = scale.clone();
                let out = sc.transform(x);
                (0..out.len()).map(|i| out.get_flat(i)).collect()
            }
        }
    }

    pub fn to_json(&self) -> String {
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str(&format!("  \"format\": \"{FORMAT}\",\n"));
        s.push_str("  \"opset\": 1,\n");
        match self {
            Self::LinearRegression { coef, intercept } => {
                s.push_str("  \"model\": {\n");
                s.push_str("    \"type\": \"LinearRegression\",\n");
                s.push_str(&format!("    \"coef\": {},\n", fmt_f64_list(coef)));
                s.push_str(&format!("    \"intercept\": {}\n", fmt_f64(*intercept)));
                s.push_str("  }\n");
            }
            Self::LogisticRegression { coef, intercept } => {
                s.push_str("  \"model\": {\n");
                s.push_str("    \"type\": \"LogisticRegression\",\n");
                s.push_str(&format!("    \"coef\": {},\n", fmt_f64_list(coef)));
                s.push_str(&format!("    \"intercept\": {}\n", fmt_f64(*intercept)));
                s.push_str("  }\n");
            }
            Self::StandardScaler { mean, scale } => {
                s.push_str("  \"model\": {\n");
                s.push_str("    \"type\": \"StandardScaler\",\n");
                s.push_str(&format!("    \"mean\": {},\n", fmt_f64_list(mean)));
                s.push_str(&format!("    \"scale\": {}\n", fmt_f64_list(scale)));
                s.push_str("  }\n");
            }
        }
        s.push('}');
        s
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let ty = extract_string(text, "\"type\"").ok_or("missing model type")?;
        match ty.as_str() {
            "LinearRegression" => Ok(Self::LinearRegression {
                coef: extract_f64_list(text, "\"coef\"")?,
                intercept: extract_f64(text, "\"intercept\"")?,
            }),
            "LogisticRegression" => Ok(Self::LogisticRegression {
                coef: extract_f64_list(text, "\"coef\"")?,
                intercept: extract_f64(text, "\"intercept\"")?,
            }),
            "StandardScaler" => Ok(Self::StandardScaler {
                mean: extract_f64_list(text, "\"mean\"")?,
                scale: extract_f64_list(text, "\"scale\"")?,
            }),
            other => Err(format!("unknown model type {other}")),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        fs::write(path, self.to_json())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
        Self::from_json(&text)
    }
}

fn fmt_f64(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else if v.is_nan() {
        "null".into()
    } else if v.is_sign_positive() {
        "1e999".into()
    } else {
        "-1e999".into()
    }
}

fn fmt_f64_list(v: &[f64]) -> String {
    let inner = v.iter().map(|&x| fmt_f64(x)).collect::<Vec<_>>().join(", ");
    format!("[{inner}]")
}

fn extract_string(text: &str, key: &str) -> Option<String> {
    let i = text.find(key)?;
    let after = &text[i + key.len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let rest = &rest[1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn extract_f64(text: &str, key: &str) -> Result<f64, String> {
    let i = text.find(key).ok_or_else(|| format!("missing {key}"))?;
    let after = &text[i + key.len()..];
    let colon = after.find(':').ok_or("bad number")?;
    let rest = after[colon + 1..].trim_start();
    let end = rest
        .find(|c: char| c == ',' || c == '}' || c == '\n')
        .unwrap_or(rest.len());
    rest[..end]
        .trim()
        .parse::<f64>()
        .map_err(|e| e.to_string())
}

fn extract_f64_list(text: &str, key: &str) -> Result<Vec<f64>, String> {
    let i = text.find(key).ok_or_else(|| format!("missing {key}"))?;
    let after = &text[i + key.len()..];
    let lb = after.find('[').ok_or("missing [")?;
    let rb = after[lb..].find(']').ok_or("missing ]")?;
    let inner = after[lb + 1..lb + rb].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|p| p.trim().parse::<f64>().map_err(|e| e.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnumpy::NdArray;

    #[test]
    fn linear_json_roundtrip_predict() {
        let x = NdArray::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0]).reshape_view(&[4, 2]);
        let y: Vec<f64> = (0..4)
            .map(|i| 1.0 + 2.0 * x.get(&[i, 0]) + 3.0 * x.get(&[i, 1]))
            .collect();
        let mut lr = LinearRegression::new();
        lr.fit(&x, &y);
        let art = ModelArtifact::from_linear(&lr);
        let json = art.to_json();
        let back = ModelArtifact::from_json(&json).unwrap();
        let p0 = lr.predict(&x);
        let p1 = back.apply(&x);
        for (a, b) in p0.iter().zip(p1.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
