//! Text feature extraction — HashingVectorizer / CountVectorizer.

use std::collections::HashMap;

use rnumpy::NdArray;

use crate::model_selection::from_shape_helper;

/// MurmurHash3 x86_32 (seed 0), returned as signed i32 like sklearn.
pub fn murmurhash3_bytes_s32(key: &[u8], seed: u32) -> i32 {
    let mut h1 = seed;
    const C1: u32 = 0xcc9e2d51;
    const C2: u32 = 0x1b873593;
    let nblocks = key.len() / 4;
    for i in 0..nblocks {
        let i4 = i * 4;
        let mut k1 = u32::from_le_bytes([key[i4], key[i4 + 1], key[i4 + 2], key[i4 + 3]]);
        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);
        h1 ^= k1;
        h1 = h1.rotate_left(13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xe6546b64);
    }
    let mut k1 = 0u32;
    let tail = &key[nblocks * 4..];
    match tail.len() {
        3 => {
            k1 ^= (tail[2] as u32) << 16;
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(C1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(C2);
            h1 ^= k1;
        }
        2 => {
            k1 ^= (tail[1] as u32) << 8;
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(C1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(C2);
            h1 ^= k1;
        }
        1 => {
            k1 ^= tail[0] as u32;
            k1 = k1.wrapping_mul(C1);
            k1 = k1.rotate_left(15);
            k1 = k1.wrapping_mul(C2);
            h1 ^= k1;
        }
        _ => {}
    }
    h1 ^= key.len() as u32;
    h1 ^= h1 >> 16;
    h1 = h1.wrapping_mul(0x85ebca6b);
    h1 ^= h1 >> 13;
    h1 = h1.wrapping_mul(0xc2b2ae35);
    h1 ^= h1 >> 16;
    h1 as i32
}

/// Word analyzer matching sklearn default: `(?u)\b\w\w+\b`, lowercased.
pub fn word_tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Skip non-word (not letter/digit/underscore). Unicode letters via char.
        let rest = &text[i..];
        let mut chars = rest.char_indices();
        let Some((start_rel, ch)) = chars.next() else {
            break;
        };
        if !is_word_char(ch) {
            i += ch.len_utf8();
            continue;
        }
        // Need word boundary before: start of string or previous not word.
        let boundary_ok = if i == 0 {
            true
        } else {
            let prev = text[..i].chars().next_back().unwrap();
            !is_word_char(prev)
        };
        if !boundary_ok {
            i += ch.len_utf8();
            continue;
        }
        let mut end = i + ch.len_utf8();
        let mut len_chars = 1usize;
        for (rel, c) in chars {
            if !is_word_char(c) {
                break;
            }
            end = i + rel + c.len_utf8();
            len_chars += 1;
            let _ = start_rel;
        }
        // Trailing boundary: end of string or next not word (already ensured by break).
        if len_chars >= 2 {
            out.push(text[i..end].to_lowercase());
        }
        i = end;
    }
    out
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `sklearn.feature_extraction.text.HashingVectorizer` (dense output, no L2 norm).
#[derive(Debug, Clone)]
pub struct HashingVectorizer {
    pub n_features: usize,
    pub alternate_sign: bool,
    pub binary: bool,
}

impl HashingVectorizer {
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            alternate_sign: true,
            binary: false,
        }
    }

    pub fn transform(&self, docs: &[String]) -> NdArray {
        let n = docs.len();
        let d = self.n_features;
        let mut data = vec![0.0; n * d];
        for (row, doc) in docs.iter().enumerate() {
            for tok in word_tokenize(doc) {
                let h = murmurhash3_bytes_s32(tok.as_bytes(), 0);
                let idx = (h.unsigned_abs() as usize) % d;
                let sign = if self.alternate_sign {
                    if h >= 0 {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    1.0
                };
                let cell = &mut data[row * d + idx];
                if self.binary {
                    *cell = sign;
                } else {
                    *cell += sign;
                }
            }
        }
        from_shape_helper(&[n, d], data)
    }

    pub fn fit_transform(&self, docs: &[String]) -> NdArray {
        self.transform(docs)
    }
}

/// `sklearn.feature_extraction.text.CountVectorizer` (dense, binary=False).
#[derive(Debug, Clone, Default)]
pub struct CountVectorizer {
    pub vocabulary_: HashMap<String, usize>,
}

impl CountVectorizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(&mut self, docs: &[String]) -> &mut Self {
        let mut vocab: Vec<String> = Vec::new();
        let mut seen = HashMap::new();
        for doc in docs {
            for tok in word_tokenize(doc) {
                if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(tok.clone()) {
                    e.insert(vocab.len());
                    vocab.push(tok);
                }
            }
        }
        vocab.sort();
        self.vocabulary_.clear();
        for (i, t) in vocab.into_iter().enumerate() {
            self.vocabulary_.insert(t, i);
        }
        self
    }

    pub fn transform(&self, docs: &[String]) -> NdArray {
        let n = docs.len();
        let d = self.vocabulary_.len();
        let mut data = vec![0.0; n * d];
        for (row, doc) in docs.iter().enumerate() {
            for tok in word_tokenize(doc) {
                if let Some(&j) = self.vocabulary_.get(&tok) {
                    data[row * d + j] += 1.0;
                }
            }
        }
        from_shape_helper(&[n, d.max(1)], if d == 0 { vec![0.0; n] } else { data })
    }

    pub fn fit_transform(&mut self, docs: &[String]) -> NdArray {
        self.fit(docs);
        self.transform(docs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn murmur_matches_sklearn_samples() {
        assert_eq!(murmurhash3_bytes_s32(b"the", 0), -1132748958);
        assert_eq!(murmurhash3_bytes_s32(b"quick", 0), 771291085);
        assert_eq!(murmurhash3_bytes_s32(b"brown", 0), 741580288);
        assert_eq!(murmurhash3_bytes_s32(b"fox", 0), -1621867415);
    }

    #[test]
    fn hashing_vectorizer_sample() {
        let docs = vec![
            "The quick brown fox".into(),
            "fox jumps over the lazy dog".into(),
            "Brown fox".into(),
        ];
        let hv = HashingVectorizer::new(16);
        let x = hv.transform(&docs);
        assert_eq!(x.get(&[0, 0]), 1.0);
        assert_eq!(x.get(&[0, 7]), -1.0);
        assert_eq!(x.get(&[0, 13]), 1.0);
        assert_eq!(x.get(&[0, 14]), -1.0);
        assert_eq!(x.get(&[1, 7]), -2.0);
        assert_eq!(x.get(&[2, 0]), 1.0);
        assert_eq!(x.get(&[2, 7]), -1.0);
    }
}
