# Core Numerical — ONNX-ish interchange (`rsklearn::export`)

Thin **JSON** dump / load for a subset of C1 estimators (`std` only). Not full ONNX
protobuf; shaped for local save/load and a tiny HTTP inference sketch.

## Layout

| Path | Role |
|------|------|
| `crates/rsklearn/src/export.rs` | `ModelArtifact` JSON serialize / deserialize / `apply` |
| `crates/rsklearn/examples/serve_onnxish.rs` | `std::net` POST `/predict` sketch |

## Supported models

| Artifact `type` | Fields | `apply` |
|-----------------|--------|---------|
| `LinearRegression` | `coef`, `intercept` | continuous predict |
| `LogisticRegression` | `coef`, `intercept` | class labels as f64 |
| `StandardScaler` | `mean`, `scale` | flattened transform |

## Format sketch

```json
{
  "format": "rsklearn-onnxish-v1",
  "opset": 1,
  "model": {
    "type": "LinearRegression",
    "coef": [2.0, 3.0],
    "intercept": 1.0
  }
}
```

## Setup

```powershell
cargo test -p rsklearn --lib
# optional HTTP sketch:
# cargo run -p rsklearn --example serve_onnxish -- model.json 8787
```

See [`ROADMAP_PHASE2.md`](ROADMAP_PHASE2.md) Track C3 and [`SKLEARN.md`](SKLEARN.md).
