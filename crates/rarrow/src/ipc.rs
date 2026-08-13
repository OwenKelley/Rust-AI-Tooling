//! Apache Arrow IPC streaming format (`std` only FlatBuffers).

use crate::array::{
    bit_is_set, validity_bitmap, Array, BooleanArray, DictionaryUtf8Array, Float64Array,
    Int64Array, ListFloat64Array, StringArray,
};
use crate::flatbuf::{
    read_offset_vector, read_string, read_struct_vector_16, root_as, table_field_bool,
    table_field_i16, table_field_i32, table_field_i64, table_field_offset,
    table_field_union_type,
};
use crate::record_batch::RecordBatch;
use crate::rev_fbb::{RevFbb, Val};
use crate::schema::{DataType, Field, Schema};

const CONTINUATION: u32 = 0xFFFF_FFFF;

// Arrow flatbuf Type union
const TYPE_INT: u8 = 2;
const TYPE_FLOATING_POINT: u8 = 3;
const TYPE_UTF8: u8 = 5;
const TYPE_BOOL: u8 = 6;
const TYPE_TIMESTAMP: u8 = 10;
const TYPE_LIST: u8 = 12;
const TIME_UNIT_NANOSECOND: i16 = 3;

// MessageHeader
const HDR_SCHEMA: u8 = 1;
const HDR_DICTIONARY_BATCH: u8 = 2;
const HDR_RECORD_BATCH: u8 = 3;

// MetadataVersion::V5
const META_V5: i16 = 4;

// Precision::DOUBLE
const PREC_DOUBLE: i16 = 2;

fn pad8(n: usize) -> usize {
    (8 - (n % 8)) % 8
}

fn align_buf(buf: &mut Vec<u8>) {
    let p = pad8(buf.len());
    buf.extend(std::iter::repeat(0).take(p));
}

fn write_i64_struct(len: i64, null_count: i64) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&len.to_le_bytes());
    b[8..].copy_from_slice(&null_count.to_le_bytes());
    b
}

fn write_buffer_struct(offset: i64, length: i64) -> [u8; 16] {
    write_i64_struct(offset, length)
}

fn encode_type(fbb: &mut RevFbb, dt: &DataType) -> (u8, u32) {
    match dt {
        DataType::Float64 => {
            let t = fbb.create_table(&[(0, Val::I16(PREC_DOUBLE))]);
            (TYPE_FLOATING_POINT, t)
        }
        DataType::Int64 => {
            let t = fbb.create_table(&[(0, Val::I32(64)), (1, Val::Bool(true))]);
            (TYPE_INT, t)
        }
        DataType::Boolean => {
            let t = fbb.create_table(&[]);
            (TYPE_BOOL, t)
        }
        DataType::Utf8 => {
            let t = fbb.create_table(&[]);
            (TYPE_UTF8, t)
        }
        DataType::TimestampNs => {
            // Timestamp: unit (i16), timezone optional string omitted.
            let t = fbb.create_table(&[(0, Val::I16(TIME_UNIT_NANOSECOND))]);
            (TYPE_TIMESTAMP, t)
        }
        DataType::ListFloat64 => {
            let t = fbb.create_table(&[]);
            (TYPE_LIST, t)
        }
        // DictionaryUtf8 is encoded as Utf8 + DictionaryEncoding on the Field.
        DataType::DictionaryUtf8 => {
            let t = fbb.create_table(&[]);
            (TYPE_UTF8, t)
        }
    }
}

fn encode_dictionary_encoding(fbb: &mut RevFbb, id: i64) -> u32 {
    let index_ty = fbb.create_table(&[(0, Val::I32(32)), (1, Val::Bool(true))]);
    // DictionaryEncoding: isOrdered → indexType → id (reverse-builder add order)
    fbb.create_table(&[
        (2, Val::Bool(false)),
        (1, Val::Offset(index_ty)),
        (0, Val::I64(id)),
    ])
}

fn encode_field(fbb: &mut RevFbb, field: &Field) -> u32 {
    encode_field_with_dict_id(fbb, field, None)
}

fn encode_field_with_dict_id(fbb: &mut RevFbb, field: &Field, dict_id: Option<i64>) -> u32 {
    let name = fbb.create_string_off(&field.name);
    let (ty, ty_off) = encode_type(fbb, &field.data_type);
    let children = if matches!(field.data_type, DataType::ListFloat64) {
        let item = Field::new("item", DataType::Float64, true);
        let child = encode_field(fbb, &item);
        fbb.create_offset_vector(&[child])
    } else {
        fbb.create_offset_vector(&[])
    };
    let dict_id = dict_id.or_else(|| {
        if matches!(field.data_type, DataType::DictionaryUtf8) {
            Some(0)
        } else {
            None
        }
    });
    if let Some(id) = dict_id {
        let dict = encode_dictionary_encoding(fbb, id);
        fbb.create_table(&[
            (5, Val::Offset(children)),
            (4, Val::Offset(dict)),
            (3, Val::Offset(ty_off)),
            (0, Val::Offset(name)),
            (2, Val::UnionType(ty)),
            (1, Val::Bool(field.nullable)),
        ])
    } else {
        fbb.create_table(&[
            (5, Val::Offset(children)),
            (3, Val::Offset(ty_off)),
            (0, Val::Offset(name)),
            (2, Val::UnionType(ty)),
            (1, Val::Bool(field.nullable)),
        ])
    }
}

fn encode_schema_table(fbb: &mut RevFbb, schema: &Schema) -> u32 {
    let mut field_offs = Vec::with_capacity(schema.fields.len());
    for (i, f) in schema.fields.iter().enumerate() {
        let dict_id = if matches!(f.data_type, DataType::DictionaryUtf8) {
            Some(i as i64)
        } else {
            None
        };
        field_offs.push(encode_field_with_dict_id(fbb, f, dict_id));
    }
    let fields_vec = fbb.create_offset_vector(&field_offs);
    fbb.create_table(&[(1, Val::Offset(fields_vec))])
}

fn encode_schema_msg(schema: &Schema) -> Vec<u8> {
    let mut fbb = RevFbb::new();
    let schema_tbl = encode_schema_table(&mut fbb, schema);
    // Message packing: header → version → header_type (matches PyArrow slots).
    let msg = fbb.create_table(&[
        (2, Val::Offset(schema_tbl)),
        (0, Val::I16(META_V5)),
        (1, Val::UnionType(HDR_SCHEMA)),
    ]);
    fbb.finish(msg)
}

fn column_buffers(col: &Array) -> (Vec<[u8; 16]>, Vec<u8>, Vec<[u8; 16]>) {
    let len = col.len() as i64;
    let nulls = col.null_count() as i64;
    let node = write_i64_struct(len, nulls);

    let mut body = Vec::new();
    let mut bufs = Vec::new();
    let mut nodes = vec![node];

    let push_buf = |body: &mut Vec<u8>, bufs: &mut Vec<[u8; 16]>, data: &[u8]| {
        let off = body.len() as i64;
        let len = data.len() as i64;
        body.extend_from_slice(data);
        align_buf(body);
        bufs.push(write_buffer_struct(off, len));
    };

    match col {
        Array::Float64(a) => {
            let vb = validity_bitmap(&a.nulls);
            push_buf(&mut body, &mut bufs, &vb);
            let mut bytes = Vec::with_capacity(a.values.len() * 8);
            for &v in &a.values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            push_buf(&mut body, &mut bufs, &bytes);
        }
        Array::Int64(a) | Array::TimestampNs(a) => {
            let vb = validity_bitmap(&a.nulls);
            push_buf(&mut body, &mut bufs, &vb);
            let mut bytes = Vec::with_capacity(a.values.len() * 8);
            for &v in &a.values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            push_buf(&mut body, &mut bufs, &bytes);
        }
        Array::Boolean(a) => {
            let vb = validity_bitmap(&a.nulls);
            push_buf(&mut body, &mut bufs, &vb);
            let n_bytes = (a.values.len() + 7) / 8;
            let mut bytes = vec![0u8; n_bytes];
            for (i, &v) in a.values.iter().enumerate() {
                if v {
                    bytes[i / 8] |= 1 << (i % 8);
                }
            }
            push_buf(&mut body, &mut bufs, &bytes);
        }
        Array::Utf8(a) => {
            let nulls: Vec<bool> = a.values.iter().map(|v| v.is_none()).collect();
            let vb = validity_bitmap(&nulls);
            push_buf(&mut body, &mut bufs, &vb);
            let mut offsets: Vec<i32> = Vec::with_capacity(a.values.len() + 1);
            let mut data = Vec::new();
            offsets.push(0);
            for v in &a.values {
                if let Some(s) = v {
                    data.extend_from_slice(s.as_bytes());
                }
                offsets.push(data.len() as i32);
            }
            let mut off_bytes = Vec::with_capacity(offsets.len() * 4);
            for o in offsets {
                off_bytes.extend_from_slice(&o.to_le_bytes());
            }
            push_buf(&mut body, &mut bufs, &off_bytes);
            push_buf(&mut body, &mut bufs, &data);
        }
        Array::ListFloat64(a) => {
            let vb = validity_bitmap(&a.nulls);
            push_buf(&mut body, &mut bufs, &vb);
            let mut off_bytes = Vec::with_capacity(a.offsets.len() * 4);
            for &o in &a.offsets {
                off_bytes.extend_from_slice(&o.to_le_bytes());
            }
            push_buf(&mut body, &mut bufs, &off_bytes);
            // Child Float64: validity (empty) + values
            let child_n = a.values.len() as i64;
            nodes.push(write_i64_struct(child_n, 0));
            push_buf(&mut body, &mut bufs, &[]); // empty child validity
            let mut bytes = Vec::with_capacity(a.values.len() * 8);
            for &v in &a.values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            push_buf(&mut body, &mut bufs, &bytes);
        }
        Array::DictionaryUtf8(a) => {
            // Indices only (dictionary values go in DictionaryBatch messages).
            let vb = validity_bitmap(&a.nulls);
            push_buf(&mut body, &mut bufs, &vb);
            let mut bytes = Vec::with_capacity(a.indices.len() * 4);
            for &v in &a.indices {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            push_buf(&mut body, &mut bufs, &bytes);
        }
    }

    (bufs, body, nodes)
}

fn encode_dictionary_batch_msg(id: i64, dictionary: &[String]) -> (Vec<u8>, Vec<u8>) {
    let dict_arr = Array::Utf8(StringArray::from_vec(
        dictionary.iter().map(|s| Some(s.clone())).collect(),
    ));
    let (bufs, col_body, nodes) = column_buffers(&dict_arr);
    let body = col_body;
    let all_bufs = bufs;
    let _ = &all_bufs;
    let mut fbb = RevFbb::new();
    let nodes_vec = fbb.create_struct_vector_16(&nodes);
    let bufs_vec = fbb.create_struct_vector_16(&all_bufs);
    let rb = fbb.create_table(&[
        (0, Val::I64(dictionary.len() as i64)),
        (2, Val::Offset(bufs_vec)),
        (1, Val::Offset(nodes_vec)),
    ]);
    // DictionaryBatch: isDelta → data → id
    let dict_batch = fbb.create_table(&[
        (2, Val::Bool(false)),
        (1, Val::Offset(rb)),
        (0, Val::I64(id)),
    ]);
    let msg = fbb.create_table(&[
        (3, Val::I64(body.len() as i64)),
        (2, Val::Offset(dict_batch)),
        (0, Val::I16(META_V5)),
        (1, Val::UnionType(HDR_DICTIONARY_BATCH)),
    ]);
    (fbb.finish(msg), body)
}

fn encode_record_batch_msg(batch: &RecordBatch) -> (Vec<u8>, Vec<u8>) {
    let mut all_bufs = Vec::new();
    let mut nodes = Vec::new();
    let mut body = Vec::new();

    for col in &batch.columns {
        let (bufs, col_body, col_nodes) = column_buffers(col);
        let base = body.len() as i64;
        for mut b in bufs {
            let off = i64::from_le_bytes(b[..8].try_into().unwrap());
            let len = i64::from_le_bytes(b[8..].try_into().unwrap());
            b[..8].copy_from_slice(&(base + off).to_le_bytes());
            b[8..].copy_from_slice(&len.to_le_bytes());
            all_bufs.push(b);
        }
        body.extend_from_slice(&col_body);
        align_buf(&mut body);
        nodes.extend(col_nodes);
    }

    let mut fbb = RevFbb::new();
    let nodes_vec = fbb.create_struct_vector_16(&nodes);
    let bufs_vec = fbb.create_struct_vector_16(&all_bufs);
    // RecordBatch packing: length → buffers → nodes (matches PyArrow slots).
    let rb = fbb.create_table(&[
        (0, Val::I64(batch.num_rows() as i64)),
        (2, Val::Offset(bufs_vec)),
        (1, Val::Offset(nodes_vec)),
    ]);
    // Message packing: bodyLength → header → version → header_type
    let msg = fbb.create_table(&[
        (3, Val::I64(body.len() as i64)),
        (2, Val::Offset(rb)),
        (0, Val::I16(META_V5)),
        (1, Val::UnionType(HDR_RECORD_BATCH)),
    ]);
    (fbb.finish(msg), body)
}

fn encapsulate(meta: &[u8], body: &[u8], out: &mut Vec<u8>) {
    // Arrow IPC: metadata_size includes FlatBuffer + padding to 8 bytes.
    let meta_pad = pad8(meta.len());
    let meta_size = meta.len() + meta_pad;
    out.extend_from_slice(&CONTINUATION.to_le_bytes());
    out.extend_from_slice(&(meta_size as i32).to_le_bytes());
    out.extend_from_slice(meta);
    out.extend(std::iter::repeat(0).take(meta_pad));
    out.extend_from_slice(body);
    let body_pad = pad8(body.len());
    out.extend(std::iter::repeat(0).take(body_pad));
}

/// Returns `(metaDataLength, bodyLength)` for a Footer `Block` (Arrow IPC file).
fn encapsulate_sized(meta: &[u8], body: &[u8], out: &mut Vec<u8>) -> (i32, i64) {
    let start = out.len();
    encapsulate(meta, body, out);
    let meta_pad = pad8(meta.len());
    let meta_size = meta.len() + meta_pad;
    let meta_data_length = (8 + meta_size) as i32;
    let body_length = body.len() as i64;
    let _ = start;
    (meta_data_length, body_length)
}

fn write_block_struct(offset: i64, meta_data_length: i32, body_length: i64) -> [u8; 24] {
    let mut b = [0u8; 24];
    b[..8].copy_from_slice(&offset.to_le_bytes());
    b[8..12].copy_from_slice(&meta_data_length.to_le_bytes());
    // 4 bytes padding for 8-byte struct alignment
    b[16..24].copy_from_slice(&body_length.to_le_bytes());
    b
}

fn encode_footer_with_dicts(
    schema: &Schema,
    dicts: &[[u8; 24]],
    batches: &[[u8; 24]],
) -> Vec<u8> {
    let mut fbb = RevFbb::new();
    let schema_tbl = encode_schema_table(&mut fbb, schema);
    let dicts_vec = fbb.create_struct_vector_24(dicts);
    let rb_vec = fbb.create_struct_vector_24(batches);
    let footer = fbb.create_table(&[
        (3, Val::Offset(rb_vec)),
        (2, Val::Offset(dicts_vec)),
        (1, Val::Offset(schema_tbl)),
        (0, Val::I16(META_V5)),
    ]);
    fbb.finish(footer)
}

const FILE_MAGIC: &[u8; 8] = b"ARROW1\0\0";
const FILE_MAGIC_END: &[u8; 6] = b"ARROW1";

/// Write Arrow IPC **file** bytes (`ARROW1` magic + footer; `pyarrow.ipc.open_file`).
pub fn write_ipc_file(batch: &RecordBatch) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(FILE_MAGIC);
    let schema_meta = encode_schema_msg(&batch.schema);
    encapsulate(&schema_meta, &[], &mut out);
    let mut dict_blocks = Vec::new();
    for (i, col) in batch.columns.iter().enumerate() {
        if let Array::DictionaryUtf8(a) = col {
            let id = batch.schema.fields[i].dict_id.unwrap_or(i as i64);
            let offset = out.len() as i64;
            let (meta, body) = encode_dictionary_batch_msg(id, &a.dictionary);
            let (md_len, body_len) = encapsulate_sized(&meta, &body, &mut out);
            dict_blocks.push(write_block_struct(offset, md_len, body_len));
        }
    }
    let rb_offset = out.len() as i64;
    let (rb_meta, body) = encode_record_batch_msg(batch);
    let (md_len, body_len) = encapsulate_sized(&rb_meta, &body, &mut out);
    let block = write_block_struct(rb_offset, md_len, body_len);
    let footer = encode_footer_with_dicts(&batch.schema, &dict_blocks, &[block]);
    out.extend_from_slice(&footer);
    out.extend_from_slice(&(footer.len() as i32).to_le_bytes());
    out.extend_from_slice(FILE_MAGIC_END);
    out
}

/// Write Arrow IPC **stream** bytes (readable by `pyarrow.ipc.open_stream`).
pub fn write_ipc_stream(batch: &RecordBatch) -> Vec<u8> {
    let mut out = Vec::new();
    let schema_meta = encode_schema_msg(&batch.schema);
    encapsulate(&schema_meta, &[], &mut out);
    for (i, col) in batch.columns.iter().enumerate() {
        if let Array::DictionaryUtf8(a) = col {
            let id = batch.schema.fields[i].dict_id.unwrap_or(i as i64);
            let (meta, body) = encode_dictionary_batch_msg(id, &a.dictionary);
            encapsulate(&meta, &body, &mut out);
        }
    }
    let (rb_meta, body) = encode_record_batch_msg(batch);
    encapsulate(&rb_meta, &body, &mut out);
    out.extend_from_slice(&CONTINUATION.to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes());
    out
}

fn parse_message(meta: &[u8]) -> (u8, usize, i64) {
    let root = root_as(meta);
    let hdr_ty = table_field_union_type(meta, root, 1).unwrap_or(0);
    let hdr = table_field_offset(meta, root, 2).expect("message header");
    let body_len = table_field_i64(meta, root, 3).unwrap_or(0);
    (hdr_ty, hdr, body_len)
}

fn decode_type(meta: &[u8], type_ty: u8, type_off: usize) -> DataType {
    match type_ty {
        TYPE_FLOATING_POINT => {
            let prec = table_field_i16(meta, type_off, 0).unwrap_or(PREC_DOUBLE);
            assert_eq!(prec, PREC_DOUBLE, "only float64 supported");
            DataType::Float64
        }
        TYPE_INT => {
            let bw = table_field_i32(meta, type_off, 0).unwrap_or(64);
            let signed = table_field_bool(meta, type_off, 1).unwrap_or(true);
            assert!(bw == 64 && signed, "only int64 supported");
            DataType::Int64
        }
        TYPE_BOOL => DataType::Boolean,
        TYPE_UTF8 => DataType::Utf8,
        TYPE_TIMESTAMP => {
            let unit = table_field_i16(meta, type_off, 0).unwrap_or(TIME_UNIT_NANOSECOND);
            assert_eq!(unit, TIME_UNIT_NANOSECOND, "only timestamp[ns] supported");
            DataType::TimestampNs
        }
        TYPE_LIST => DataType::ListFloat64,
        other => panic!("unsupported Arrow type union tag {other}"),
    }
}

fn decode_schema(meta: &[u8], schema_tbl: usize) -> Schema {
    let fields_off = table_field_offset(meta, schema_tbl, 1).expect("schema.fields");
    let field_tables = read_offset_vector(meta, fields_off);
    let mut fields = Vec::new();
    for ft in field_tables {
        let name = table_field_offset(meta, ft, 0)
            .map(|p| read_string(meta, p).to_string())
            .unwrap_or_default();
        let nullable = table_field_bool(meta, ft, 1).unwrap_or(false);
        let type_ty = table_field_union_type(meta, ft, 2).expect("field type_type");
        let type_off = table_field_offset(meta, ft, 3).expect("field type");
        let mut dt = decode_type(meta, type_ty, type_off);
        let mut dict_id = None;
        if let Some(dict_off) = table_field_offset(meta, ft, 4) {
            dict_id = table_field_i64(meta, dict_off, 0);
            if matches!(dt, DataType::Utf8) {
                dt = DataType::DictionaryUtf8;
            }
        }
        if matches!(dt, DataType::ListFloat64) {
            if let Some(ch_off) = table_field_offset(meta, ft, 5) {
                let children = read_offset_vector(meta, ch_off);
                if let Some(&cft) = children.first() {
                    let cty = table_field_union_type(meta, cft, 2).unwrap_or(0);
                    let coff = table_field_offset(meta, cft, 3).unwrap_or(0);
                    let child_dt = decode_type(meta, cty, coff);
                    assert!(
                        matches!(child_dt, DataType::Float64),
                        "only list<float64> supported"
                    );
                }
            }
        }
        let mut field = Field::new(name, dt, nullable);
        field.dict_id = dict_id;
        fields.push(field);
    }
    Schema::new(fields)
}

fn read_buffers_body<'a>(body: &'a [u8], bufs: &[[u8; 16]]) -> Vec<&'a [u8]> {
    bufs.iter()
        .map(|b| {
            let off = i64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
            let len = i64::from_le_bytes(b[8..].try_into().unwrap()) as usize;
            &body[off..off + len]
        })
        .collect()
}

fn decode_column(dt: &DataType, n: usize, null_count: usize, bufs: &[&[u8]], bi: &mut usize) -> Array {
    match dt {
        DataType::Float64 => {
            let validity = bufs[*bi];
            *bi += 1;
            let values = bufs[*bi];
            *bi += 1;
            let mut nulls = vec![false; n];
            if null_count > 0 && !validity.is_empty() {
                for i in 0..n {
                    nulls[i] = !bit_is_set(validity, i);
                }
            }
            let mut vals = Vec::with_capacity(n);
            for i in 0..n {
                let p = i * 8;
                vals.push(f64::from_le_bytes(values[p..p + 8].try_into().unwrap()));
            }
            Array::Float64(Float64Array {
                values: vals,
                nulls,
            })
        }
        DataType::Int64 | DataType::TimestampNs => {
            let validity = bufs[*bi];
            *bi += 1;
            let values = bufs[*bi];
            *bi += 1;
            let mut nulls = vec![false; n];
            if null_count > 0 && !validity.is_empty() {
                for i in 0..n {
                    nulls[i] = !bit_is_set(validity, i);
                }
            }
            let mut vals = Vec::with_capacity(n);
            for i in 0..n {
                let p = i * 8;
                vals.push(i64::from_le_bytes(values[p..p + 8].try_into().unwrap()));
            }
            let arr = Int64Array {
                values: vals,
                nulls,
            };
            if matches!(dt, DataType::TimestampNs) {
                Array::TimestampNs(arr)
            } else {
                Array::Int64(arr)
            }
        }
        DataType::Boolean => {
            let validity = bufs[*bi];
            *bi += 1;
            let values = bufs[*bi];
            *bi += 1;
            let mut nulls = vec![false; n];
            if null_count > 0 && !validity.is_empty() {
                for i in 0..n {
                    nulls[i] = !bit_is_set(validity, i);
                }
            }
            let mut vals = Vec::with_capacity(n);
            for i in 0..n {
                vals.push(bit_is_set(values, i));
            }
            Array::Boolean(BooleanArray {
                values: vals,
                nulls,
            })
        }
        DataType::Utf8 => {
            let validity = bufs[*bi];
            *bi += 1;
            let offsets = bufs[*bi];
            *bi += 1;
            let data = bufs[*bi];
            *bi += 1;
            let nulls_present = null_count > 0 && !validity.is_empty();
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let is_null = nulls_present && !bit_is_set(validity, i);
                if is_null {
                    out.push(None);
                } else {
                    let start = i32::from_le_bytes(offsets[i * 4..i * 4 + 4].try_into().unwrap())
                        as usize;
                    let end = i32::from_le_bytes(
                        offsets[(i + 1) * 4..(i + 1) * 4 + 4]
                            .try_into()
                            .unwrap(),
                    ) as usize;
                    out.push(Some(
                        String::from_utf8(data[start..end].to_vec()).expect("utf8"),
                    ));
                }
            }
            Array::Utf8(StringArray { values: out })
        }
        DataType::ListFloat64 => {
            let validity = bufs[*bi];
            *bi += 1;
            let offsets_b = bufs[*bi];
            *bi += 1;
            let mut offsets = Vec::with_capacity(n + 1);
            for i in 0..=n {
                offsets.push(i32::from_le_bytes(
                    offsets_b[i * 4..i * 4 + 4].try_into().unwrap(),
                ));
            }
            let mut nulls = vec![false; n];
            if null_count > 0 && !validity.is_empty() {
                for i in 0..n {
                    nulls[i] = !bit_is_set(validity, i);
                }
            }
            // Child float64: validity + values (consumed even if empty validity)
            let _child_validity = bufs[*bi];
            *bi += 1;
            let values_b = bufs[*bi];
            *bi += 1;
            let child_n = offsets.last().copied().unwrap_or(0) as usize;
            let mut values = Vec::with_capacity(child_n);
            for i in 0..child_n {
                let p = i * 8;
                values.push(f64::from_le_bytes(values_b[p..p + 8].try_into().unwrap()));
            }
            Array::ListFloat64(ListFloat64Array {
                offsets,
                values,
                nulls,
            })
        }
        DataType::DictionaryUtf8 => {
            let validity = bufs[*bi];
            *bi += 1;
            let values = bufs[*bi];
            *bi += 1;
            let mut nulls = vec![false; n];
            if null_count > 0 && !validity.is_empty() {
                for i in 0..n {
                    nulls[i] = !bit_is_set(validity, i);
                }
            }
            let mut indices = Vec::with_capacity(n);
            for i in 0..n {
                let p = i * 4;
                indices.push(i32::from_le_bytes(values[p..p + 4].try_into().unwrap()));
            }
            // Dictionary filled by caller.
            Array::DictionaryUtf8(DictionaryUtf8Array {
                indices,
                nulls,
                dictionary: Vec::new(),
            })
        }
    }
}

fn decode_record_batch(
    meta: &[u8],
    rb_tbl: usize,
    body: &[u8],
    schema: &Schema,
    dictionaries: &std::collections::HashMap<i64, Vec<String>>,
) -> RecordBatch {
    let length = table_field_i64(meta, rb_tbl, 0).unwrap_or(0) as usize;
    let nodes_off = table_field_offset(meta, rb_tbl, 1).expect("nodes");
    let bufs_off = table_field_offset(meta, rb_tbl, 2).expect("buffers");
    let nodes = read_struct_vector_16(meta, nodes_off);
    let buf_descs = read_struct_vector_16(meta, bufs_off);
    let bufs = read_buffers_body(body, &buf_descs);
    let mut bi = 0;
    let mut ni = 0;
    let mut columns = Vec::new();
    for (fi, field) in schema.fields.iter().enumerate() {
        let null_count = i64::from_le_bytes(nodes[ni][8..].try_into().unwrap()) as usize;
        let n = i64::from_le_bytes(nodes[ni][..8].try_into().unwrap()) as usize;
        assert_eq!(n, length);
        ni += 1;
        if matches!(field.data_type, DataType::ListFloat64) {
            ni += 1;
        }
        let mut col = decode_column(
            &field.data_type,
            length,
            null_count,
            &bufs,
            &mut bi,
        );
        if let Array::DictionaryUtf8(ref mut a) = col {
            let id = field.dict_id.unwrap_or(fi as i64);
            a.dictionary = dictionaries
                .get(&id)
                .cloned()
                .unwrap_or_default();
        }
        columns.push(col);
    }
    assert_eq!(ni, nodes.len());
    RecordBatch::try_new(schema.clone(), columns)
}

/// Read Arrow IPC **stream** bytes (from `write_ipc_stream` or `pyarrow.ipc.new_stream`).
pub fn read_ipc_stream(bytes: &[u8]) -> RecordBatch {
    let mut pos = 0;
    let mut schema: Option<Schema> = None;
    let mut batch: Option<RecordBatch> = None;
    let mut dictionaries: std::collections::HashMap<i64, Vec<String>> =
        std::collections::HashMap::new();

    while pos + 8 <= bytes.len() {
        let cont = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        assert_eq!(cont, CONTINUATION, "bad IPC continuation");
        let msize = i32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap());
        pos += 8;
        if msize == 0 {
            break; // EOS
        }
        let meta = &bytes[pos..pos + msize as usize];
        pos += msize as usize;
        // If metadata_size already includes padding (multiple of 8), this is 0.
        pos += pad8(pos);

        let (hdr_ty, hdr, body_len) = parse_message(meta);
        let body = &bytes[pos..pos + body_len as usize];
        pos += body_len as usize;
        pos += pad8(pos);

        match hdr_ty {
            HDR_SCHEMA => {
                schema = Some(decode_schema(meta, hdr));
            }
            HDR_DICTIONARY_BATCH => {
                let id = table_field_i64(meta, hdr, 0).unwrap_or(0);
                let rb_off = table_field_offset(meta, hdr, 1).expect("dictionary data");
                let dict_schema = Schema::new(vec![Field::new("dict", DataType::Utf8, true)]);
                let empty = std::collections::HashMap::new();
                let dict_batch = decode_record_batch(meta, rb_off, body, &dict_schema, &empty);
                let strings = match &dict_batch.columns[0] {
                    Array::Utf8(a) => a
                        .values
                        .iter()
                        .map(|v| v.clone().unwrap_or_default())
                        .collect(),
                    _ => panic!("dictionary values must be utf8"),
                };
                dictionaries.insert(id, strings);
            }
            HDR_RECORD_BATCH => {
                let sch = schema.as_ref().expect("RecordBatch before Schema");
                batch = Some(decode_record_batch(
                    meta,
                    hdr,
                    body,
                    sch,
                    &dictionaries,
                ));
            }
            other => panic!("unsupported message header {other}"),
        }
    }

    batch.expect("no RecordBatch in stream")
}

fn read_struct_vector_24(buf: &[u8], pos: usize) -> Vec<[u8; 24]> {
    let n = i32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(n);
    let mut p = pos + 4;
    for _ in 0..n {
        let mut e = [0u8; 24];
        e.copy_from_slice(&buf[p..p + 24]);
        out.push(e);
        p += 24;
    }
    out
}

/// Read Arrow IPC **file** bytes (`write_ipc_file` or `pyarrow.ipc.new_file`).
pub fn read_ipc_file(bytes: &[u8]) -> RecordBatch {
    assert!(bytes.len() >= 8 + 10, "IPC file truncated");
    assert_eq!(&bytes[..8], FILE_MAGIC, "bad IPC file magic");
    assert_eq!(
        &bytes[bytes.len() - 6..],
        FILE_MAGIC_END,
        "bad IPC file end magic"
    );
    let footer_len =
        i32::from_le_bytes(bytes[bytes.len() - 10..bytes.len() - 6].try_into().unwrap()) as usize;
    let footer_start = bytes.len() - 10 - footer_len;
    let footer = &bytes[footer_start..footer_start + footer_len];
    let root = root_as(footer);
    let schema_tbl = table_field_offset(footer, root, 1).expect("footer.schema");
    let schema = decode_schema(footer, schema_tbl);
    let mut dictionaries: std::collections::HashMap<i64, Vec<String>> =
        std::collections::HashMap::new();
    if let Some(dicts_off) = table_field_offset(footer, root, 2) {
        let dict_blocks = read_struct_vector_24(footer, dicts_off);
        for block in &dict_blocks {
            let offset = i64::from_le_bytes(block[..8].try_into().unwrap()) as usize;
            let meta_data_length = i32::from_le_bytes(block[8..12].try_into().unwrap()) as usize;
            let body_length = i64::from_le_bytes(block[16..24].try_into().unwrap()) as usize;
            let cont = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            assert_eq!(cont, CONTINUATION);
            let msize = i32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
            let meta = &bytes[offset + 8..offset + 8 + msize];
            let body_start = offset + meta_data_length;
            let body = &bytes[body_start..body_start + body_length];
            let (hdr_ty, hdr, _) = parse_message(meta);
            assert_eq!(hdr_ty, HDR_DICTIONARY_BATCH);
            let id = table_field_i64(meta, hdr, 0).unwrap_or(0);
            let rb_off = table_field_offset(meta, hdr, 1).expect("dictionary data");
            let dict_schema = Schema::new(vec![Field::new("dict", DataType::Utf8, true)]);
            let empty = std::collections::HashMap::new();
            let dict_batch = decode_record_batch(meta, rb_off, body, &dict_schema, &empty);
            let strings = match &dict_batch.columns[0] {
                Array::Utf8(a) => a
                    .values
                    .iter()
                    .map(|v| v.clone().unwrap_or_default())
                    .collect(),
                _ => panic!("dictionary values must be utf8"),
            };
            dictionaries.insert(id, strings);
        }
    }
    let batches_off = table_field_offset(footer, root, 3).expect("footer.recordBatches");
    let blocks = read_struct_vector_24(footer, batches_off);
    assert!(!blocks.is_empty(), "IPC file has no record batches");
    let block = &blocks[0];
    let offset = i64::from_le_bytes(block[..8].try_into().unwrap()) as usize;
    let meta_data_length = i32::from_le_bytes(block[8..12].try_into().unwrap()) as usize;
    let body_length = i64::from_le_bytes(block[16..24].try_into().unwrap()) as usize;
    assert!(offset + meta_data_length + body_length <= bytes.len());
    let cont = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    assert_eq!(cont, CONTINUATION);
    let msize = i32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
    let meta = &bytes[offset + 8..offset + 8 + msize];
    assert_eq!(8 + msize, meta_data_length, "block metaDataLength mismatch");
    let body_start = offset + meta_data_length;
    let body = &bytes[body_start..body_start + body_length];
    let (hdr_ty, hdr, body_len) = parse_message(meta);
    assert_eq!(hdr_ty, HDR_RECORD_BATCH);
    assert_eq!(body_len as usize, body_length);
    decode_record_batch(meta, hdr, body, &schema, &dictionaries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{Float64Array, Int64Array, StringArray};
    use crate::record_batch::batch_from_columns;

    #[test]
    fn ipc_roundtrip_mixed() {
        let batch = batch_from_columns(vec![
            (
                "a".into(),
                Array::Float64(Float64Array::from_slice(&[1.0, 2.5, -3.0])),
            ),
            (
                "b".into(),
                Array::Int64(Int64Array::from_nullable(&[Some(1), None, Some(3)])),
            ),
            (
                "d".into(),
                Array::Utf8(StringArray::from_vec(vec![
                    Some("x".into()),
                    None,
                    Some("yz".into()),
                ])),
            ),
        ]);
        let bytes = write_ipc_stream(&batch);
        let back = read_ipc_stream(&bytes);
        assert_eq!(back.num_rows(), 3);
        assert_eq!(back.checksum(), batch.checksum());
    }

    #[test]
    fn read_pyarrow_mixed_stream() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../target/ref_mixed.stream"
        );
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return, // optional fixture
        };
        let batch = read_ipc_stream(&bytes);
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 4);
        // a: 1+2.5-3=0.5; b: 1+3=4; c: 1+0+1=2;
        // d: "x" → 1+120=121; "yz" → 2+121+122=245
        let expected = 3.0 + 4.0 + 0.5 + 4.0 + 2.0 + 121.0 + 245.0;
        assert!(
            (batch.checksum() - expected).abs() < 1e-9,
            "got {}",
            batch.checksum()
        );
    }

    #[test]
    fn write_stream_for_pyarrow_fixture() {
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
        let bytes = write_ipc_stream(&batch);
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../target/rarrow_out.stream"
        );
        let _ = std::fs::write(path, &bytes);
    }

    #[test]
    fn ipc_file_roundtrip() {
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
        let bytes = write_ipc_file(&batch);
        assert_eq!(&bytes[..8], b"ARROW1\0\0");
        assert_eq!(&bytes[bytes.len() - 6..], b"ARROW1");
        let back = read_ipc_file(&bytes);
        assert_eq!(back.checksum(), batch.checksum());
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../target/rarrow_out.arrow"
        );
        let _ = std::fs::write(path, &bytes);
    }

    #[test]
    fn read_pyarrow_ipc_file() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../target/ref_mixed.arrow"
        );
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return,
        };
        let batch = read_ipc_file(&bytes);
        assert!(batch.num_rows() > 0);
    }

    #[test]
    fn ipc_timestamp_and_list_roundtrip() {
        use crate::array::ListFloat64Array;
        let batch = batch_from_columns(vec![
            (
                "ts".into(),
                Array::TimestampNs(Int64Array::from_slice(&[
                    1_700_000_000_000_000_000,
                    1_700_000_000_000_000_001,
                ])),
            ),
            (
                "xs".into(),
                Array::ListFloat64(ListFloat64Array::from_slices(&[
                    vec![1.0, 2.0],
                    vec![3.0],
                ])),
            ),
        ]);
        let bytes = write_ipc_stream(&batch);
        let back = read_ipc_stream(&bytes);
        assert_eq!(back.num_rows(), 2);
        assert_eq!(back.checksum(), batch.checksum());
        assert!(matches!(
            back.schema.fields[0].data_type,
            crate::schema::DataType::TimestampNs
        ));
        assert!(matches!(
            back.schema.fields[1].data_type,
            crate::schema::DataType::ListFloat64
        ));
    }

    #[test]
    fn ipc_dictionary_utf8_roundtrip() {
        use crate::array::DictionaryUtf8Array;
        let batch = batch_from_columns(vec![(
            "cat".into(),
            Array::DictionaryUtf8(DictionaryUtf8Array::from_values(&[
                Some("red"),
                Some("blue"),
                Some("red"),
                None,
                Some("blue"),
            ])),
        )]);
        assert_eq!(
            match &batch.columns[0] {
                Array::DictionaryUtf8(a) => a.dictionary.len(),
                _ => 0,
            },
            2
        );
        let bytes = write_ipc_stream(&batch);
        let back = read_ipc_stream(&bytes);
        assert_eq!(back.checksum(), batch.checksum());
        assert!(matches!(
            back.schema.fields[0].data_type,
            crate::schema::DataType::DictionaryUtf8
        ));
        let file = write_ipc_file(&batch);
        let back_f = read_ipc_file(&file);
        assert_eq!(back_f.checksum(), batch.checksum());
    }
}
