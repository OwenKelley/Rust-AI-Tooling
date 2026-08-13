# Core Numerical — scikit-learn parity (Python ↔ Rust)

Maps **scikit-learn**-shaped classical ML onto `rsklearn` (`std` only, on `rnumpy`).

## Layout

| Path | Role |
|------|------|
| `crates/rsklearn` | Split, scaler, linear/logistic, k-NN, k-means, metrics, text vectorizers |
| `python/core_numerical/sklearn_parity` | Reference ops + compare harness |
| `crates/parity_runner` bin `sklearn_parity_runner` | Rust timings / checksums |

## API map

| Python (`sklearn`) | Rust (`rsklearn`) |
|--------------------|-------------------|
| `train_test_split` | `rsklearn::train_test_split` |
| `StandardScaler` / `LabelEncoder` | same names |
| `LinearRegression` / `LogisticRegression` | same |
| `KNeighborsClassifier` / `Regressor` | same |
| `KMeans` | same |
| `accuracy_score`, `precision/recall/f1`, `mse/mae/r2` | same |
| `HashingVectorizer` / `CountVectorizer` | `feature_extraction::{HashingVectorizer, CountVectorizer}` |
| ONNX-ish JSON dump | `rsklearn::ModelArtifact` ([`ONNXISH.md`](ONNXISH.md)) |

## Setup

```powershell
cargo test -p rsklearn
cargo build -p parity_runner --bin sklearn_parity_runner --release
cd python
python -m core_numerical.sklearn_parity.compare --size 32 --iters 5 --warmup 1
```

## Notes

- `train_test_split(..., shuffle=False)`: train = first rows, test = last; `n_test = ceil(test_size * n)` (sklearn).
- Logistic parity compares **accuracy** (GD vs sklearn LBFGS); coefs need not match.
- KMeans parity uses fixed `init=X[:k]` and Lloyd; inertia compared.
- `HashingVectorizer` uses murmurhash3 + word analyzer (`\b\w\w+\b`, lowercase) like sklearn.

See [`ROADMAP_PHASE2.md`](ROADMAP_PHASE2.md) Track C1 / C2 / C3.
