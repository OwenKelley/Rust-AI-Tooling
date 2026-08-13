//! Minimal Apache Parquet (`PAR1`) PLAIN / UNCOMPRESSED for f64 / i64 / bool / utf8.
//! Also retains the legacy `RPQT` IPC container for Rust↔Rust roundtrips.

use crate::array::{bit_is_set, Array, BooleanArray, Float64Array, Int64Array, StringArray};
use crate::ipc::{read_ipc_stream, write_ipc_stream};
use crate::record_batch::RecordBatch;
use crate::schema::{DataType, Field, Schema};

const RPQT_MAGIC: &[u8; 4] = b"RPQT";
const RPQT_VERSION: u32 = 1;
const PAR1_MAGIC: &[u8; 4] = b"PAR1";

// parquet.thrift enums
const TYPE_BOOLEAN: i32 = 0;
const TYPE_INT64: i32 = 2;
const TYPE_DOUBLE: i32 = 5;
const TYPE_BYTE_ARRAY: i32 = 6;

const CONVERTED_UTF8: i32 = 0;

const ENC_PLAIN: i32 = 0;
const ENC_RLE: i32 = 3;

const PAGE_DATA: i32 = 0;
const COMPRESSION_UNCOMPRESSED: i32 = 0;

// --- Thrift compact protocol (write) -------------------------------------------------

fn zigzag_i32(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

fn write_varint(out: &mut Vec<u8>, mut n: u32) {
    while n >= 0x80 {
        out.push((n as u8) | 0x80);
        n >>= 7;
    }
    out.push(n as u8);
}

fn write_field_begin(out: &mut Vec<u8>, last_id: &mut i16, field_id: i16, typ: u8) {
    let delta = field_id - *last_id;
    if delta > 0 && delta <= 15 {
        out.push(((delta as u8) << 4) | typ);
    } else {
        out.push(typ);
        // zigzag field id as i16 varint
        let z = zigzag_i32(field_id as i32);
        write_varint(out, z);
    }
    *last_id = field_id;
}

fn write_i32_field(out: &mut Vec<u8>, last_id: &mut i16, field_id: i16, v: i32) {
    write_field_begin(out, last_id, field_id, 5); // I32
    write_varint(out, zigzag_i32(v));
}

fn write_i64_field(out: &mut Vec<u8>, last_id: &mut i16, field_id: i16, v: i64) {
    write_field_begin(out, last_id, field_id, 6); // I64
    let z = ((v << 1) ^ (v >> 63)) as u64;
    let mut n = z;
    while n >= 0x80 {
        out.push((n as u8) | 0x80);
        n >>= 7;
    }
    out.push(n as u8);
}

fn write_binary_field(out: &mut Vec<u8>, last_id: &mut i16, field_id: i16, bytes: &[u8]) {
    write_field_begin(out, last_id, field_id, 8); // BINARY
    write_varint(out, bytes.len() as u32);
    out.extend_from_slice(bytes);
}

fn write_struct_begin(_out: &mut Vec<u8>) {}
fn write_struct_end(out: &mut Vec<u8>) {
    out.push(0); // STOP
}

fn write_list_begin(out: &mut Vec<u8>, last_id: &mut i16, field_id: i16, elem_type: u8, n: usize) {
    write_field_begin(out, last_id, field_id, 9); // LIST
    if n < 15 {
        out.push(((n as u8) << 4) | elem_type);
    } else {
        out.push(0xf0 | elem_type);
        write_varint(out, n as u32);
    }
}

// elem types: STRUCT=12, I32=5, etc.
const T_STRUCT: u8 = 12;
const T_I32: u8 = 5;

fn page_header(num_values: i32, page_size: i32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut last = 0i16;
    write_struct_begin(&mut out);
    write_i32_field(&mut out, &mut last, 1, PAGE_DATA);
    write_i32_field(&mut out, &mut last, 2, page_size);
    write_i32_field(&mut out, &mut last, 3, page_size);
    write_field_begin(&mut out, &mut last, 5, T_STRUCT);
    {
        let mut inner = 0i16;
        write_i32_field(&mut out, &mut inner, 1, num_values);
        write_i32_field(&mut out, &mut inner, 2, ENC_PLAIN);
        write_i32_field(&mut out, &mut inner, 3, ENC_RLE);
        write_i32_field(&mut out, &mut inner, 4, ENC_RLE);
        // empty Statistics struct (field 5) — matches PyArrow
        write_field_begin(&mut out, &mut inner, 5, T_STRUCT);
        write_struct_end(&mut out);
        write_struct_end(&mut out);
    }
    write_struct_end(&mut out);
    out
}

/// RLE/bitpack hybrid definition levels for max_def=1 (nullable).
/// Parquet: bit-packed header LSB=1, RLE header LSB=0. No bitwidth byte (implied).
fn def_levels_rle(nulls: &[bool]) -> Vec<u8> {
    let n = nulls.len();
    let mut rle = Vec::new();
    let mut i = 0;
    while i < n {
        let present = !nulls[i];
        let mut run = 1usize;
        while i + run < n && (!nulls[i + run]) == present {
            run += 1;
        }
        // Prefer RLE for runs (header = run_len << 1, LSB=0).
        let header = (run as u32) << 1;
        write_varint(&mut rle, header);
        rle.push(u8::from(present));
        i += run;
    }
    let mut out = Vec::with_capacity(4 + rle.len());
    out.extend_from_slice(&(rle.len() as i32).to_le_bytes());
    out.extend_from_slice(&rle);
    out
}

fn plain_encode(col: &Array) -> (Vec<u8>, Vec<bool>, i32) {
    match col {
        Array::Float64(a) => {
            let mut vals = Vec::new();
            for (i, &v) in a.values.iter().enumerate() {
                if !a.nulls[i] {
                    vals.extend_from_slice(&v.to_le_bytes());
                }
            }
            (vals, a.nulls.clone(), a.values.len() as i32)
        }
        Array::Int64(a) | Array::TimestampNs(a) => {
            let mut vals = Vec::new();
            for (i, &v) in a.values.iter().enumerate() {
                if !a.nulls[i] {
                    vals.extend_from_slice(&v.to_le_bytes());
                }
            }
            (vals, a.nulls.clone(), a.values.len() as i32)
        }
        Array::Boolean(a) => {
            let present: Vec<bool> = a
                .values
                .iter()
                .zip(a.nulls.iter())
                .filter_map(|(&v, &n)| if n { None } else { Some(v) })
                .collect();
            let n_bytes = (present.len() + 7) / 8;
            let mut bytes = vec![0u8; n_bytes];
            for (i, v) in present.iter().enumerate() {
                if *v {
                    bytes[i / 8] |= 1 << (i % 8);
                }
            }
            (bytes, a.nulls.clone(), a.values.len() as i32)
        }
        Array::Utf8(a) => {
            let mut vals = Vec::new();
            let mut nulls = Vec::with_capacity(a.values.len());
            for v in &a.values {
                match v {
                    None => nulls.push(true),
                    Some(s) => {
                        nulls.push(false);
                        let b = s.as_bytes();
                        vals.extend_from_slice(&(b.len() as i32).to_le_bytes());
                        vals.extend_from_slice(b);
                    }
                }
            }
            (vals, nulls, a.values.len() as i32)
        }
        Array::ListFloat64(_) => {
            panic!("parquet: ListFloat64 not supported in PAR1 v1 (use IPC)")
        }
        Array::DictionaryUtf8(a) => {
            // Densify to PLAIN utf8 for parquet.
            let mut vals = Vec::new();
            let mut nulls = Vec::with_capacity(a.indices.len());
            for (i, &idx) in a.indices.iter().enumerate() {
                if a.nulls[i] {
                    nulls.push(true);
                } else {
                    nulls.push(false);
                    let s = &a.dictionary[idx as usize];
                    let b = s.as_bytes();
                    vals.extend_from_slice(&(b.len() as i32).to_le_bytes());
                    vals.extend_from_slice(b);
                }
            }
            (vals, nulls, a.indices.len() as i32)
        }
    }
}

fn parquet_type(dt: &DataType) -> (i32, Option<i32>) {
    match dt {
        DataType::Boolean => (TYPE_BOOLEAN, None),
        DataType::Int64 | DataType::TimestampNs => (TYPE_INT64, None),
        DataType::Float64 => (TYPE_DOUBLE, None),
        DataType::Utf8 | DataType::DictionaryUtf8 => (TYPE_BYTE_ARRAY, Some(CONVERTED_UTF8)),
        DataType::ListFloat64 => panic!("parquet: ListFloat64 not supported in PAR1 v1"),
    }
}

fn write_column_chunk(out: &mut Vec<u8>, col: &Array, field: &Field) -> (i64, i64, i64) {
    let data_page_offset = out.len() as i64;
    let (plain, nulls, num_values) = plain_encode(col);
    let mut page_body = Vec::new();
    if field.nullable {
        page_body.extend_from_slice(&def_levels_rle(&nulls));
    }
    page_body.extend_from_slice(&plain);
    let header = page_header(num_values, page_body.len() as i32);
    out.extend_from_slice(&header);
    out.extend_from_slice(&page_body);
    let total = (out.len() as i64) - data_page_offset;
    (data_page_offset, total, data_page_offset)
}

fn schema_element(name: &str, typ: Option<i32>, converted: Option<i32>, rep: i32, num_children: i32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut last = 0i16;
    write_struct_begin(&mut out);
    if let Some(t) = typ {
        write_i32_field(&mut out, &mut last, 1, t);
    }
    // 3: repetition_type, 4: name, 5: num_children, 6: converted_type
    write_i32_field(&mut out, &mut last, 3, rep);
    write_binary_field(&mut out, &mut last, 4, name.as_bytes());
    if num_children > 0 {
        write_i32_field(&mut out, &mut last, 5, num_children);
    }
    if let Some(c) = converted {
        write_i32_field(&mut out, &mut last, 6, c);
    }
    write_struct_end(&mut out);
    out
}

/// Serialize as Apache Parquet `PAR1` (PLAIN / UNCOMPRESSED).
pub fn write_parquet_par1(batch: &RecordBatch) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(PAR1_MAGIC);

    let mut chunk_metas: Vec<(String, i32, Option<i32>, i64, i64, i64, i32, bool)> = Vec::new();

    for (field, col) in batch.schema.fields.iter().zip(batch.columns.iter()) {
        let (ty, conv) = parquet_type(&field.data_type);
        let start = out.len() as i64;
        let (_fo, size, dpo) = write_column_chunk(&mut out, col, field);
        let _ = _fo;
        chunk_metas.push((
            field.name.clone(),
            ty,
            conv,
            start,
            size,
            dpo,
            col.len() as i32,
            field.nullable,
        ));
    }

    let num_rows = batch.num_rows() as i64;
    let mut footer = Vec::new();
    let mut last = 0i16;
    write_struct_begin(&mut footer);
    write_i32_field(&mut footer, &mut last, 1, 1); // version
    {
        let mut selems = Vec::new();
        let mut root = Vec::new();
        let mut rl = 0i16;
        // Root: name + num_children only (no type / repetition)
        write_binary_field(&mut root, &mut rl, 4, b"schema");
        write_i32_field(&mut root, &mut rl, 5, batch.num_columns() as i32);
        write_struct_end(&mut root);
        selems.push(root);
        for field in &batch.schema.fields {
            let (ty, conv) = parquet_type(&field.data_type);
            let rep = if field.nullable { 1 } else { 0 }; // OPTIONAL=1 REQUIRED=0
            selems.push(schema_element(&field.name, Some(ty), conv, rep, 0));
        }
        write_list_begin(&mut footer, &mut last, 2, T_STRUCT, selems.len());
        for s in selems {
            footer.extend_from_slice(&s);
        }
    }
    write_i64_field(&mut footer, &mut last, 3, num_rows);
    write_list_begin(&mut footer, &mut last, 4, T_STRUCT, 1);
    {
        let mut rg_last = 0i16;
        write_list_begin(&mut footer, &mut rg_last, 1, T_STRUCT, chunk_metas.len());
        for (name, ty, _conv, file_off, size, dpo, nvals, _nullable) in &chunk_metas {
            let mut cl = 0i16;
            // ColumnChunk: 2 file_offset, 3 meta_data
            write_i64_field(&mut footer, &mut cl, 2, *file_off);
            write_field_begin(&mut footer, &mut cl, 3, T_STRUCT);
            {
                let mut m = 0i16;
                write_i32_field(&mut footer, &mut m, 1, *ty);
                // encodings used in pages
                write_list_begin(&mut footer, &mut m, 2, T_I32, 2);
                write_varint(&mut footer, zigzag_i32(ENC_PLAIN));
                write_varint(&mut footer, zigzag_i32(ENC_RLE));
                write_list_begin(&mut footer, &mut m, 3, 8, 1);
                write_varint(&mut footer, name.len() as u32);
                footer.extend_from_slice(name.as_bytes());
                write_i32_field(&mut footer, &mut m, 4, COMPRESSION_UNCOMPRESSED);
                write_i64_field(&mut footer, &mut m, 5, *nvals as i64);
                write_i64_field(&mut footer, &mut m, 6, *size);
                write_i64_field(&mut footer, &mut m, 7, *size);
                write_i64_field(&mut footer, &mut m, 9, *dpo);
                write_struct_end(&mut footer);
            }
            write_struct_end(&mut footer);
        }
        // RowGroup: 1 columns, 2 total_byte_size, 3 num_rows
        write_i64_field(&mut footer, &mut rg_last, 2, (out.len() as i64) - 4);
        write_i64_field(&mut footer, &mut rg_last, 3, num_rows);
        write_struct_end(&mut footer);
    }
    write_struct_end(&mut footer);

    let flen = footer.len() as i32;
    out.extend_from_slice(&footer);
    out.extend_from_slice(&flen.to_le_bytes());
    out.extend_from_slice(PAR1_MAGIC);
    out
}

/// Serialize a record batch. Prefers Apache `PAR1`; use [`write_parquet_rpqt`] for legacy.
pub fn write_parquet(batch: &RecordBatch) -> Vec<u8> {
    write_parquet_par1(batch)
}

/// Legacy RPQT container (IPC payload).
pub fn write_parquet_rpqt(batch: &RecordBatch) -> Vec<u8> {
    let ipc = write_ipc_stream(batch);
    let mut out = Vec::with_capacity(4 + 4 + 8 + ipc.len());
    out.extend_from_slice(RPQT_MAGIC);
    out.extend_from_slice(&RPQT_VERSION.to_le_bytes());
    out.extend_from_slice(&(ipc.len() as u64).to_le_bytes());
    out.extend_from_slice(&ipc);
    out
}

/// Deserialize RPQT or Apache `PAR1` (dtype subset).
pub fn read_parquet(bytes: &[u8]) -> RecordBatch {
    assert!(bytes.len() >= 8, "parquet truncated");
    if &bytes[..4] == RPQT_MAGIC {
        return read_parquet_rpqt(bytes);
    }
    if &bytes[..4] == PAR1_MAGIC {
        return read_parquet_par1(bytes);
    }
    panic!("expected RPQT or PAR1 magic");
}

fn read_parquet_rpqt(bytes: &[u8]) -> RecordBatch {
    assert_eq!(&bytes[..4], RPQT_MAGIC);
    let ver = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    assert_eq!(ver, RPQT_VERSION, "unsupported RPQT version {ver}");
    let n = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    assert_eq!(bytes.len(), 16 + n, "RPQT length mismatch");
    read_ipc_stream(&bytes[16..])
}

// --- Thrift compact (read, minimal) -------------------------------------------------

fn read_varint(data: &[u8], pos: &mut usize) -> u32 {
    let mut shift = 0u32;
    let mut result = 0u32;
    loop {
        let b = data[*pos];
        *pos += 1;
        result |= u32::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    result
}

fn zagzig_i32(n: u32) -> i32 {
    ((n >> 1) as i32) ^ -((n & 1) as i32)
}

fn read_parquet_par1(bytes: &[u8]) -> RecordBatch {
    assert_eq!(&bytes[..4], PAR1_MAGIC);
    assert_eq!(&bytes[bytes.len() - 4..], PAR1_MAGIC);
    let flen = i32::from_le_bytes(bytes[bytes.len() - 8..bytes.len() - 4].try_into().unwrap()) as usize;
    let footer = &bytes[bytes.len() - 8 - flen..bytes.len() - 8];
    // Parse schema names + types and row group column chunk offsets via a loose scan.
    // For v1 we rely on writing our own layout and reading it back; also accept pyarrow
    // files for the same dtype subset by using pyarrow-compatible chunk offsets from footer.
    let parsed = parse_footer_chunks(footer);
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    for ch in &parsed.chunks {
        let page = &bytes[ch.data_page_offset as usize..];
        let (arr, dt, nullable) = decode_column_page(page, ch.physical_type, ch.num_values as usize);
        fields.push(Field::new(ch.name.clone(), dt, nullable));
        columns.push(arr);
    }
    RecordBatch::try_new(Schema::new(fields), columns)
}

struct ChunkMeta {
    name: String,
    physical_type: i32,
    num_values: i64,
    data_page_offset: i64,
}

struct FooterInfo {
    chunks: Vec<ChunkMeta>,
}

fn parse_footer_chunks(footer: &[u8]) -> FooterInfo {
    let mut pos = 0usize;
    let chunks = walk_file_metadata(footer, &mut pos);
    FooterInfo { chunks }
}

struct ThriftField {
    id: i16,
    typ: u8,
    start: usize,
}

fn read_struct_fields<'a>(data: &'a [u8], pos: &mut usize) -> Vec<ThriftField> {
    let mut last_id = 0i16;
    let mut fields = Vec::new();
    loop {
        let header = data[*pos];
        *pos += 1;
        if header == 0 {
            break;
        }
        let typ = header & 0x0f;
        let delta = header >> 4;
        let id = if delta == 0 {
            let z = read_varint(data, pos);
            zagzig_i32(z) as i16
        } else {
            last_id + delta as i16
        };
        last_id = id;
        let start = *pos;
        skip_value(data, pos, typ);
        fields.push(ThriftField {
            id,
            typ,
            start,
        });
    }
    fields
}

fn skip_value(data: &[u8], pos: &mut usize, typ: u8) {
    match typ {
        1 | 2 => {} // BOOL true/false — value in type nibble for some; compact uses types 1/2 as true/false with no payload
        3 => {
            // bool alternative
        }
        4 => {
            // BYTE — actually I16?
            let _ = read_varint(data, pos);
        }
        5 => {
            let _ = read_varint(data, pos);
        } // I32
        6 => {
            let _ = read_varint64(data, pos);
        } // I64
        7 => {
            *pos += 8;
        } // DOUBLE
        8 => {
            let n = read_varint(data, pos) as usize;
            *pos += n;
        } // BINARY
        9 | 10 => {
            // LIST / SET
            let header = data[*pos];
            *pos += 1;
            let elem = header & 0x0f;
            let mut n = (header >> 4) as usize;
            if n == 15 {
                n = read_varint(data, pos) as usize;
            }
            for _ in 0..n {
                skip_value(data, pos, elem);
            }
        }
        11 => {
            // MAP
            let size_and = data[*pos];
            *pos += 1;
            let mut n = (size_and >> 4) as usize;
            if n == 15 {
                n = read_varint(data, pos) as usize;
            }
            let key_t = size_and & 0x0f;
            let val_t = data[*pos];
            *pos += 1;
            for _ in 0..n {
                skip_value(data, pos, key_t);
                skip_value(data, pos, val_t);
            }
        }
        12 => {
            let _ = read_struct_fields(data, pos);
        }
        other => panic!("unsupported thrift type {other}"),
    }
}

fn read_varint64(data: &[u8], pos: &mut usize) -> u64 {
    let mut shift = 0u32;
    let mut result = 0u64;
    loop {
        let b = data[*pos];
        *pos += 1;
        result |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    result
}

fn walk_file_metadata(data: &[u8], pos: &mut usize) -> Vec<ChunkMeta> {
    let fields = read_struct_fields(data, pos);
    let mut chunks = Vec::new();
    // field 4 = row_groups
    for f in &fields {
        if f.id == 4 && f.typ == 9 {
            let mut p = f.start;
            let header = data[p];
            p += 1;
            let elem = header & 0x0f;
            let mut n = (header >> 4) as usize;
            if n == 15 {
                n = read_varint(data, &mut p) as usize;
            }
            assert_eq!(elem, T_STRUCT);
            for _ in 0..n {
                chunks.extend(walk_row_group(data, &mut p));
            }
        }
    }
    chunks
}

fn walk_row_group(data: &[u8], pos: &mut usize) -> Vec<ChunkMeta> {
    let fields = read_struct_fields(data, pos);
    let mut chunks = Vec::new();
    for f in &fields {
        if f.id == 1 && f.typ == 9 {
            let mut p = f.start;
            let header = data[p];
            p += 1;
            let elem = header & 0x0f;
            let mut n = (header >> 4) as usize;
            if n == 15 {
                n = read_varint(data, &mut p) as usize;
            }
            assert_eq!(elem, T_STRUCT);
            for _ in 0..n {
                if let Some(c) = walk_column_chunk(data, &mut p) {
                    chunks.push(c);
                }
            }
        }
    }
    chunks
}

fn walk_column_chunk(data: &[u8], pos: &mut usize) -> Option<ChunkMeta> {
    let fields = read_struct_fields(data, pos);
    let mut meta: Option<ChunkMeta> = None;
    for f in &fields {
        if f.id == 3 && f.typ == 12 {
            let mut p = f.start;
            meta = Some(walk_column_meta(data, &mut p));
        }
    }
    meta
}

fn walk_column_meta(data: &[u8], pos: &mut usize) -> ChunkMeta {
    let fields = read_struct_fields(data, pos);
    let mut name = String::new();
    let mut physical_type = TYPE_DOUBLE;
    let mut num_values = 0i64;
    let mut data_page_offset = 0i64;
    for f in &fields {
        let mut p = f.start;
        match f.id {
            1 if f.typ == 5 => {
                physical_type = zagzig_i32(read_varint(data, &mut p));
            }
            3 if f.typ == 9 => {
                // path_in_schema list of binary
                let header = data[p];
                p += 1;
                let mut n = (header >> 4) as usize;
                if n == 15 {
                    n = read_varint(data, &mut p) as usize;
                }
                for i in 0..n {
                    let len = read_varint(data, &mut p) as usize;
                    let s = std::str::from_utf8(&data[p..p + len]).unwrap_or("");
                    if i == 0 {
                        name = s.to_string();
                    } else {
                        name.push('.');
                        name.push_str(s);
                    }
                    p += len;
                }
            }
            5 if f.typ == 6 => {
                let z = read_varint64(data, &mut p);
                num_values = ((z >> 1) as i64) ^ -((z & 1) as i64);
            }
            9 if f.typ == 6 => {
                let z = read_varint64(data, &mut p);
                data_page_offset = ((z >> 1) as i64) ^ -((z & 1) as i64);
            }
            _ => {}
        }
    }
    ChunkMeta {
        name,
        physical_type,
        num_values,
        data_page_offset,
    }
}

fn decode_column_page(page: &[u8], physical_type: i32, num_values: usize) -> (Array, DataType, bool) {
    let mut pos = 0usize;
    let fields = read_struct_fields(page, &mut pos);
    let mut uncompressed = 0i32;
    let mut has_data_header = false;
    for f in &fields {
        if f.id == 2 && f.typ == 5 {
            let mut p = f.start;
            uncompressed = zagzig_i32(read_varint(page, &mut p));
        }
        if f.id == 5 {
            has_data_header = true;
        }
    }
    let _ = has_data_header;
    let body = &page[pos..pos + uncompressed as usize];
    // Detect def levels: if first 4 bytes look like a length and bitwidth follows
    let (values_bytes, nulls, nullable) = split_def_and_values(body, num_values);
    let arr = match physical_type {
        TYPE_DOUBLE => {
            let mut vals = vec![0.0; num_values];
            let mut vi = 0usize;
            for i in 0..num_values {
                if !nulls[i] {
                    vals[i] = f64::from_le_bytes(values_bytes[vi..vi + 8].try_into().unwrap());
                    vi += 8;
                }
            }
            Array::Float64(Float64Array {
                values: vals,
                nulls: nulls.clone(),
            })
        }
        TYPE_INT64 => {
            let mut vals = vec![0; num_values];
            let mut vi = 0usize;
            for i in 0..num_values {
                if !nulls[i] {
                    vals[i] = i64::from_le_bytes(values_bytes[vi..vi + 8].try_into().unwrap());
                    vi += 8;
                }
            }
            Array::Int64(Int64Array {
                values: vals,
                nulls: nulls.clone(),
            })
        }
        TYPE_BOOLEAN => {
            let mut vals = vec![false; num_values];
            let mut bit_i = 0usize;
            for i in 0..num_values {
                if !nulls[i] {
                    vals[i] = bit_is_set(values_bytes, bit_i);
                    bit_i += 1;
                }
            }
            Array::Boolean(BooleanArray {
                values: vals,
                nulls: nulls.clone(),
            })
        }
        TYPE_BYTE_ARRAY => {
            let mut out = Vec::with_capacity(num_values);
            let mut vi = 0usize;
            for i in 0..num_values {
                if nulls[i] {
                    out.push(None);
                } else {
                    let len = i32::from_le_bytes(values_bytes[vi..vi + 4].try_into().unwrap()) as usize;
                    vi += 4;
                    let s = String::from_utf8(values_bytes[vi..vi + len].to_vec()).expect("utf8");
                    vi += len;
                    out.push(Some(s));
                }
            }
            Array::Utf8(StringArray { values: out })
        }
        other => panic!("unsupported parquet physical type {other}"),
    };
    let dt = match physical_type {
        TYPE_DOUBLE => DataType::Float64,
        TYPE_INT64 => DataType::Int64,
        TYPE_BOOLEAN => DataType::Boolean,
        TYPE_BYTE_ARRAY => DataType::Utf8,
        _ => unreachable!(),
    };
    (arr, dt, nullable)
}

fn split_def_and_values(body: &[u8], num_values: usize) -> (&[u8], Vec<bool>, bool) {
    if body.len() >= 5 {
        let def_len = i32::from_le_bytes(body[..4].try_into().unwrap()) as usize;
        if def_len > 0 && def_len + 4 <= body.len() {
            let rle = &body[4..4 + def_len];
            if let Some(nulls) = decode_def_levels(rle, num_values) {
                return (&body[4 + def_len..], nulls, true);
            }
        }
    }
    (body, vec![false; num_values], false)
}

fn decode_def_levels(rle: &[u8], num_values: usize) -> Option<Vec<bool>> {
    let mut pos = 0usize;
    let mut nulls = Vec::with_capacity(num_values);
    while nulls.len() < num_values && pos < rle.len() {
        let header = read_varint(rle, &mut pos);
        if header & 1 == 0 {
            // RLE (LSB=0): count << 1
            let run = (header >> 1) as usize;
            if pos >= rle.len() {
                return None;
            }
            let present = rle[pos] != 0;
            pos += 1;
            for _ in 0..run {
                if nulls.len() >= num_values {
                    break;
                }
                nulls.push(!present);
            }
        } else {
            // Bit-packed (LSB=1): groups-of-8 << 1 | 1
            let groups = (header >> 1) as usize;
            for _ in 0..groups {
                if pos >= rle.len() {
                    return None;
                }
                let byte = rle[pos];
                pos += 1;
                for bit in 0..8 {
                    if nulls.len() >= num_values {
                        break;
                    }
                    nulls.push(byte & (1 << bit) == 0);
                }
            }
        }
    }
    if nulls.len() != num_values {
        return None;
    }
    Some(nulls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{Array, Float64Array, Int64Array};
    use crate::record_batch::batch_from_columns;

    #[test]
    fn parquet_rpqt_roundtrip() {
        let batch = batch_from_columns(vec![(
            "x".into(),
            Array::Float64(Float64Array::from_slice(&[1.0, 2.0])),
        )]);
        let b = write_parquet_rpqt(&batch);
        assert_eq!(&b[..4], b"RPQT");
        let back = read_parquet(&b);
        assert_eq!(back.checksum(), batch.checksum());
    }

    #[test]
    fn parquet_par1_roundtrip_mixed() {
        let batch = batch_from_columns(vec![
            (
                "a".into(),
                Array::Float64(Float64Array::from_slice(&[1.0, 2.5, -3.0])),
            ),
            (
                "b".into(),
                Array::Int64(Int64Array::from_nullable(&[Some(1), None, Some(3)])),
            ),
        ]);
        let b = write_parquet_par1(&batch);
        assert_eq!(&b[..4], b"PAR1");
        assert_eq!(&b[b.len() - 4..], b"PAR1");
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../target/rarrow_out.parquet"
        );
        let _ = std::fs::write(path, &b);
        let back = read_parquet(&b);
        assert_eq!(back.num_rows(), 3);
        assert_eq!(back.checksum(), batch.checksum());
    }
}
