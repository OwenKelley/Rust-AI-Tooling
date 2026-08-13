//! `rtokenizers` — thin Whitespace / BPE / WordPiece encode-decode surface (`std` only).

use std::collections::HashMap;

/// Encoded sequence: token ids and optional surface forms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoding {
    pub ids: Vec<u32>,
    pub tokens: Vec<String>,
}

/// Whitespace tokenizer: split on Unicode whitespace, map via vocab (UNK = 0 if missing).
#[derive(Debug, Clone, Default)]
pub struct WhitespaceTokenizer {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
    pub unk_id: u32,
}

impl WhitespaceTokenizer {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            id_to_token: Vec::new(),
            unk_id: 0,
        }
    }

    /// Build vocab from documents (sorted unique tokens). Reserves id 0 for `[UNK]`.
    pub fn fit(&mut self, texts: &[&str]) -> &mut Self {
        let mut set: Vec<String> = Vec::new();
        let mut seen = HashMap::new();
        for t in texts {
            for tok in t.split_whitespace() {
                if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(tok.to_string()) {
                    e.insert(());
                    set.push(tok.to_string());
                }
            }
        }
        set.sort();
        self.vocab.clear();
        self.id_to_token.clear();
        self.id_to_token.push("[UNK]".into());
        self.vocab.insert("[UNK]".into(), 0);
        self.unk_id = 0;
        for (i, tok) in set.into_iter().enumerate() {
            let id = (i + 1) as u32;
            self.vocab.insert(tok.clone(), id);
            self.id_to_token.push(tok);
        }
        self
    }

    pub fn encode(&self, text: &str) -> Encoding {
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        for tok in text.split_whitespace() {
            let id = self.vocab.get(tok).copied().unwrap_or(self.unk_id);
            ids.push(id);
            tokens.push(tok.to_string());
        }
        Encoding { ids, tokens }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| {
                self.id_to_token
                    .get(id as usize)
                    .map(|s| s.as_str())
                    .unwrap_or("[UNK]")
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Minimal BPE: bytes/chars as base, apply merges in order (greedy leftmost longest).
#[derive(Debug, Clone, Default)]
pub struct BpeTokenizer {
    pub merges: Vec<(String, String)>,
    pub vocab: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
    pub unk_id: u32,
}

impl BpeTokenizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// `merges` lines are `"a b"` pairs in merge order. Vocab = all symbols seen + merges.
    pub fn from_merges(merges: &[(&str, &str)]) -> Self {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut id_to_token = Vec::new();
        let push = |s: String, vocab: &mut HashMap<String, u32>, id_to: &mut Vec<String>| {
            if let std::collections::hash_map::Entry::Vacant(e) = vocab.entry(s.clone()) {
                let id = id_to.len() as u32;
                e.insert(id);
                id_to.push(s);
            }
        };
        push("[UNK]".into(), &mut vocab, &mut id_to_token);
        let mut merge_list = Vec::new();
        for &(a, b) in merges {
            push(a.into(), &mut vocab, &mut id_to_token);
            push(b.into(), &mut vocab, &mut id_to_token);
            let merged = format!("{a}{b}");
            push(merged, &mut vocab, &mut id_to_token);
            merge_list.push((a.to_string(), b.to_string()));
        }
        Self {
            merges: merge_list,
            vocab,
            id_to_token,
            unk_id: 0,
        }
    }

    fn apply_bpe(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return Vec::new();
        }
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        for (a, b) in &self.merges {
            let mut i = 0;
            while i + 1 < symbols.len() {
                if &symbols[i] == a && &symbols[i + 1] == b {
                    let merged = format!("{a}{b}");
                    symbols[i] = merged;
                    symbols.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
        symbols
    }

    pub fn encode(&self, text: &str) -> Encoding {
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        for word in text.split_whitespace() {
            for piece in self.apply_bpe(word) {
                let id = self.vocab.get(&piece).copied().unwrap_or(self.unk_id);
                ids.push(id);
                tokens.push(piece);
            }
        }
        Encoding { ids, tokens }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| {
                self.id_to_token
                    .get(id as usize)
                    .map(|s| s.as_str())
                    .unwrap_or("[UNK]")
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

/// WordPiece: greedy longest-match from left; continuation pieces use `##` prefix.
#[derive(Debug, Clone, Default)]
pub struct WordPieceTokenizer {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
    pub unk_id: u32,
    pub max_input_chars_per_word: usize,
}

impl WordPieceTokenizer {
    pub fn from_vocab(tokens: &[&str]) -> Self {
        let mut vocab = HashMap::new();
        let mut id_to_token = Vec::new();
        for &t in tokens {
            if let std::collections::hash_map::Entry::Vacant(e) = vocab.entry(t.to_string()) {
                let id = id_to_token.len() as u32;
                e.insert(id);
                id_to_token.push(t.to_string());
            }
        }
        let unk_id = vocab.get("[UNK]").copied().unwrap_or(0);
        Self {
            vocab,
            id_to_token,
            unk_id,
            max_input_chars_per_word: 100,
        }
    }

    pub fn encode(&self, text: &str) -> Encoding {
        let mut ids = Vec::new();
        let mut tokens = Vec::new();
        for word in text.split_whitespace() {
            if word.chars().count() > self.max_input_chars_per_word {
                ids.push(self.unk_id);
                tokens.push("[UNK]".into());
                continue;
            }
            let chars: Vec<char> = word.chars().collect();
            let mut start = 0usize;
            let mut ok = true;
            let mut word_tokens = Vec::new();
            while start < chars.len() {
                let mut end = chars.len();
                let mut cur = None;
                while start < end {
                    let substr: String = chars[start..end].iter().collect();
                    let candidate = if start > 0 {
                        format!("##{substr}")
                    } else {
                        substr
                    };
                    if self.vocab.contains_key(&candidate) {
                        cur = Some(candidate);
                        break;
                    }
                    end -= 1;
                }
                if cur.is_none() {
                    ok = false;
                    break;
                }
                word_tokens.push(cur.unwrap());
                start = end;
            }
            if !ok {
                ids.push(self.unk_id);
                tokens.push("[UNK]".into());
            } else {
                for t in word_tokens {
                    ids.push(*self.vocab.get(&t).unwrap());
                    tokens.push(t);
                }
            }
        }
        Encoding { ids, tokens }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            let t = self
                .id_to_token
                .get(id as usize)
                .map(|s| s.as_str())
                .unwrap_or("[UNK]");
            if let Some(rest) = t.strip_prefix("##") {
                out.push_str(rest);
            } else {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(t);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_roundtrip() {
        let mut tok = WhitespaceTokenizer::new();
        tok.fit(&["hello world", "hello rust"]);
        let enc = tok.encode("hello world");
        assert_eq!(enc.tokens, vec!["hello", "world"]);
        assert_eq!(tok.decode(&enc.ids), "hello world");
    }

    #[test]
    fn bpe_merges() {
        let bpe = BpeTokenizer::from_merges(&[("a", "b"), ("ab", "c")]);
        let enc = bpe.encode("abc");
        assert_eq!(enc.tokens, vec!["abc"]);
        assert_eq!(bpe.decode(&enc.ids), "abc");
    }

    #[test]
    fn wordpiece_greedy() {
        let wp = WordPieceTokenizer::from_vocab(&["[UNK]", "un", "##want", "##ed", "want"]);
        let enc = wp.encode("unwanted");
        assert_eq!(enc.tokens, vec!["un", "##want", "##ed"]);
        assert_eq!(wp.decode(&enc.ids), "unwanted");
    }
}
