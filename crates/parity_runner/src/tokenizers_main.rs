//! CLI for rtokenizers parity harness.

use std::env;
use std::process;
use std::time::Instant;

use rtokenizers::{BpeTokenizer, WhitespaceTokenizer, WordPieceTokenizer};

#[derive(Debug, Clone)]
enum Op {
    Whitespace,
    Bpe,
    WordPiece,
}

impl Op {
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "whitespace" => Self::Whitespace,
            "bpe" => Self::Bpe,
            "wordpiece" => Self::WordPiece,
            other => return Err(format!("unknown op '{other}'")),
        })
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Whitespace => "whitespace",
            Self::Bpe => "bpe",
            Self::WordPiece => "wordpiece",
        }
    }
}

struct Args {
    op: Op,
    size: usize,
    seed: u64,
    iters: usize,
    warmup: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut op = None;
    let mut size = 64usize;
    let mut seed = 42u64;
    let mut iters = 20usize;
    let mut warmup = 3usize;
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--op" => {
                op = Some(Op::parse(
                    &args.next().ok_or_else(|| "--op needs value".to_string())?,
                )?);
            }
            "--size" => {
                size = args
                    .next()
                    .ok_or_else(|| "--size needs value".to_string())?
                    .parse()
                    .map_err(|e| format!("size: {e}"))?;
            }
            "--seed" => {
                seed = args
                    .next()
                    .ok_or_else(|| "--seed needs value".to_string())?
                    .parse()
                    .map_err(|e| format!("seed: {e}"))?;
            }
            "--iters" => {
                iters = args
                    .next()
                    .ok_or_else(|| "--iters needs value".to_string())?
                    .parse()
                    .map_err(|e| format!("iters: {e}"))?;
            }
            "--warmup" => {
                warmup = args
                    .next()
                    .ok_or_else(|| "--warmup needs value".to_string())?
                    .parse()
                    .map_err(|e| format!("warmup: {e}"))?;
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    Ok(Args {
        op: op.ok_or_else(|| "missing --op".to_string())?,
        size,
        seed,
        iters,
        warmup,
    })
}

fn make_texts(n: usize, seed: u64) -> Vec<String> {
    let words = ["ab", "abc", "unwanted", "hello", "world", "rust", "code"];
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let a = words[(state as usize) % words.len()];
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let b = words[(state as usize) % words.len()];
            format!("{a} {b}")
        })
        .collect()
}

fn checksum_ids(ids: &[u32]) -> f64 {
    ids.len() as f64 + ids.iter().map(|&x| x as f64).sum::<f64>()
}

fn run_op(op: &Op, size: usize, seed: u64) -> (f64, Box<dyn FnMut()>) {
    let n = size.max(8);
    let texts = make_texts(n, seed);
    match op {
        Op::Whitespace => {
            let corpus: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let mut tok = WhitespaceTokenizer::new();
            tok.fit(&corpus);
            let mut sum = 0.0;
            for t in &texts {
                sum += checksum_ids(&tok.encode(t).ids);
            }
            (
                sum,
                Box::new(move || {
                    let corpus: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                    let mut tok = WhitespaceTokenizer::new();
                    tok.fit(&corpus);
                    for t in &texts {
                        std::hint::black_box(tok.encode(t));
                    }
                }),
            )
        }
        Op::Bpe => {
            let bpe = BpeTokenizer::from_merges(&[("a", "b"), ("ab", "c")]);
            let mut sum = 0.0;
            for t in &texts {
                sum += checksum_ids(&bpe.encode(t).ids);
            }
            (
                sum,
                Box::new(move || {
                    let bpe = BpeTokenizer::from_merges(&[("a", "b"), ("ab", "c")]);
                    for t in &texts {
                        std::hint::black_box(bpe.encode(t));
                    }
                }),
            )
        }
        Op::WordPiece => {
            let wp = WordPieceTokenizer::from_vocab(&[
                "[UNK]", "un", "##want", "##ed", "want", "hello", "world", "rust", "code", "ab",
                "abc",
            ]);
            let mut sum = 0.0;
            for t in &texts {
                sum += checksum_ids(&wp.encode(t).ids);
            }
            (
                sum,
                Box::new(move || {
                    let wp = WordPieceTokenizer::from_vocab(&[
                        "[UNK]", "un", "##want", "##ed", "want", "hello", "world", "rust", "code",
                        "ab", "abc",
                    ]);
                    for t in &texts {
                        std::hint::black_box(wp.encode(t));
                    }
                }),
            )
        }
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(2);
        }
    };
    let (checksum, mut thunk) = run_op(&args.op, args.size, args.seed);
    for _ in 0..args.warmup {
        thunk();
    }
    let mut times = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        let t0 = Instant::now();
        thunk();
        times.push(t0.elapsed().as_nanos() as u64);
    }
    times.sort_unstable();
    let median = times[times.len() / 2];
    println!(
        "{{\"op\":\"{}\",\"checksum\":{checksum},\"median_ns\":{median},\"iters\":{}}}",
        args.op.as_str(),
        args.iters
    );
}
