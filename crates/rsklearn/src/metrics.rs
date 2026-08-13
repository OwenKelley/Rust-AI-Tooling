//! Classification / regression metrics.

pub fn accuracy_score(y_true: &[i64], y_pred: &[i64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len());
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| a == b)
        .count();
    correct as f64 / n as f64
}

pub fn precision_score(y_true: &[i64], y_pred: &[i64], pos_label: i64) -> f64 {
    let mut tp = 0usize;
    let mut fp = 0usize;
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if p == pos_label {
            if t == pos_label {
                tp += 1;
            } else {
                fp += 1;
            }
        }
    }
    if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    }
}

pub fn recall_score(y_true: &[i64], y_pred: &[i64], pos_label: i64) -> f64 {
    let mut tp = 0usize;
    let mut fn_ = 0usize;
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if t == pos_label {
            if p == pos_label {
                tp += 1;
            } else {
                fn_ += 1;
            }
        }
    }
    if tp + fn_ == 0 {
        0.0
    } else {
        tp as f64 / (tp + fn_) as f64
    }
}

pub fn f1_score(y_true: &[i64], y_pred: &[i64], pos_label: i64) -> f64 {
    let p = precision_score(y_true, y_pred, pos_label);
    let r = recall_score(y_true, y_pred, pos_label);
    if p + r == 0.0 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    }
}

pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len());
    let n = y_true.len() as f64;
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f64>()
        / n
}

pub fn mean_absolute_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len());
    let n = y_true.len() as f64;
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / n
}

pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len());
    let n = y_true.len() as f64;
    let mean = y_true.iter().sum::<f64>() / n;
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum();
    let ss_tot: f64 = y_true
        .iter()
        .map(|a| {
            let d = a - mean;
            d * d
        })
        .sum();
    if ss_tot == 0.0 {
        0.0
    } else {
        1.0 - ss_res / ss_tot
    }
}
