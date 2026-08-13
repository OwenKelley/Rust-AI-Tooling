# Core Numerical — tokenizers parity (Python ↔ Rust)

Thin **Whitespace / BPE / WordPiece** surface in `rtokenizers` (`std` only).
Not a Hugging Face `tokenizers` FFI wrap; algorithms are intentionally small and
parity-tested against a matching Python reference under `tokenizers_parity`.

## Layout

| Path | Role |
|------|------|
| `crates/rtokenizers` | WhitespaceTokenizer, BpeTokenizer, WordPieceTokenizer |
| `crates/parity_runner` bin `tokenizers_parity_runner` | Rust timings / checksums |
| `python/core_numerical/tokenizers_parity` | Mirror reference + compare |

## API map

| Concept | Rust (`rtokenizers`) |
|---------|----------------------|
| Whitespace split + vocab ids | `WhitespaceTokenizer::{fit,encode,decode}` |
| BPE merges (greedy left-to-right) | `BpeTokenizer::from_merges` + `encode` / `decode` |
| WordPiece longest-match (`##`) | `WordPieceTokenizer::from_vocab` + `encode` / `decode` |
| Encoding payload | `Encoding { ids, tokens }` |

Classical text features for sklearn-shaped pipelines live in `rsklearn`
(`HashingVectorizer`, `CountVectorizer`) — see [`SKLEARN.md`](SKLEARN.md).

## Setup

```powershell
cargo test -p rtokenizers
cargo build -p parity_runner --bin tokenizers_parity_runner --release
cd python
python -m core_numerical.tokenizers_parity.compare --size 32 --iters 5 --warmup 1
```

See [`ROADMAP_PHASE2.md`](ROADMAP_PHASE2.md) Track C2.
