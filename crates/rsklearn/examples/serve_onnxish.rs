//! Minimal HTTP/JSON inference sketch over `std::net` (no external deps).
//!
//! ```text
//! cargo run -p rsklearn --example serve_onnxish -- model.json 8787
//! curl -X POST http://127.0.0.1:8787/predict -d "{\"x\":[[1.0,0.0],[0.0,1.0]]}"
//! ```

use std::env;
use std::io::{Read, Write};
use std::net::TcpListener;

use rnumpy::NdArray;
use rsklearn::export::ModelArtifact;

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        eprintln!("usage: serve_onnxish <model.json> [port]");
        std::process::exit(2);
    });
    let port: u16 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8787);
    let model = ModelArtifact::load(&path).unwrap_or_else(|e| {
        eprintln!("load model: {e}");
        std::process::exit(1);
    });
    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind");
    eprintln!("listening on http://127.0.0.1:{port}/predict");
    for stream in listener.incoming().flatten() {
        let mut stream = stream;
        let mut buf = [0u8; 8192];
        let n = stream.read(&mut buf).unwrap_or(0);
        let req = String::from_utf8_lossy(&buf[..n]);
        let body = req
            .split("\r\n\r\n")
            .nth(1)
            .unwrap_or("")
            .trim_end_matches('\0');
        let response = match handle(&model, body) {
            Ok(json) => format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                json.len(),
                json
            ),
            Err(e) => {
                let msg = format!("{{\"error\":\"{e}\"}}");
                format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    msg.len(),
                    msg
                )
            }
        };
        let _ = stream.write_all(response.as_bytes());
    }
}

fn handle(model: &ModelArtifact, body: &str) -> Result<String, String> {
    let x = parse_x(body)?;
    let y = model.apply(&x);
    Ok(format!(
        "{{\"y\":[{}]}}",
        y.iter()
            .map(|v| format!("{v}"))
            .collect::<Vec<_>>()
            .join(",")
    ))
}

fn parse_x(body: &str) -> Result<NdArray, String> {
    let key = "\"x\"";
    let i = body.find(key).ok_or("missing x")?;
    let after = &body[i + key.len()..];
    let start = after.find('[').ok_or("missing [")?;
    // Find matching outer array end by depth.
    let bytes = after.as_bytes();
    let mut depth = 0i32;
    let mut end = None;
    for (j, &b) in bytes[start..].iter().enumerate() {
        match b {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(start + j);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end.ok_or("unclosed x")?;
    let matrix = &after[start..=end];
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut inner = matrix.trim();
    if inner.starts_with('[') && inner.ends_with(']') {
        inner = &inner[1..inner.len() - 1];
    }
    // Split top-level rows.
    let mut depth = 0i32;
    let mut start_row = 0usize;
    let chars: Vec<char> = inner.chars().collect();
    let mut parts = Vec::new();
    for (i, &c) in chars.iter().enumerate() {
        match c {
            '[' => depth += 1,
            ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(chars[start_row..i].iter().collect::<String>());
                start_row = i + 1;
            }
            _ => {}
        }
    }
    if start_row < chars.len() {
        parts.push(chars[start_row..].iter().collect::<String>());
    }
    for part in parts {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let p = p.trim_matches(|c| c == '[' || c == ']');
        let row: Result<Vec<f64>, _> = p
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<f64>())
            .collect();
        rows.push(row.map_err(|e| e.to_string())?);
    }
    if rows.is_empty() {
        return Err("empty x".into());
    }
    let d = rows[0].len();
    let n = rows.len();
    let mut flat = Vec::with_capacity(n * d);
    for r in &rows {
        if r.len() != d {
            return Err("ragged rows".into());
        }
        flat.extend_from_slice(r);
    }
    Ok(NdArray::from_vec(flat).reshape_view(&[n, d]))
}
