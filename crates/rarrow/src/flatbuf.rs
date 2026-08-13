//! Forward FlatBuffers builder with absolute offsets (Arrow IPC metadata).

#[derive(Clone, Debug)]
pub enum Val {
    U8(u8),
    Bool(bool),
    I16(i16),
    I32(i32),
    I64(i64),
    Offset(u32),
    UnionType(u8),
}

pub struct Fbb {
    pub buf: Vec<u8>,
}

impl Fbb {
    pub fn new() -> Self {
        // Reserve 8 bytes for root uoffset + pad so alignments never shift.
        Self {
            buf: vec![0u8; 8],
        }
    }

    pub fn align(&mut self, a: usize) {
        while self.buf.len() % a != 0 {
            self.buf.push(0);
        }
    }

    pub fn create_string(&mut self, s: &str) -> u32 {
        self.align(4);
        let pos = self.buf.len() as u32;
        self.buf.extend_from_slice(&(s.len() as i32).to_le_bytes());
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0);
        self.align(4);
        pos
    }

    pub fn create_struct_vector_16(&mut self, elems: &[[u8; 16]]) -> u32 {
        while (self.buf.len() + 4) % 8 != 0 {
            self.buf.push(0);
        }
        let pos = self.buf.len() as u32;
        self.buf.extend_from_slice(&(elems.len() as i32).to_le_bytes());
        for e in elems {
            self.buf.extend_from_slice(e);
        }
        pos
    }

    pub fn create_offset_vector(&mut self, offsets: &[u32]) -> u32 {
        self.align(4);
        let pos = self.buf.len() as u32;
        self.buf.extend_from_slice(&(offsets.len() as i32).to_le_bytes());
        let base = self.buf.len();
        self.buf.resize(base + offsets.len() * 4, 0);
        for (i, &off) in offsets.iter().enumerate() {
            let slot = base + i * 4;
            let rel = off as i32 - slot as i32;
            self.buf[slot..slot + 4].copy_from_slice(&rel.to_le_bytes());
        }
        pos
    }

    pub fn create_table(&mut self, fields: &[(u16, Val)]) -> u32 {
        let max_id = fields.iter().map(|(id, _)| *id).max().unwrap_or(0);
        let nslots = max_id as usize + 1;
        let mut slot_off = vec![0i16; nslots];
        let mut inline = Vec::new();
        let mut offset_fixups: Vec<(usize, u32)> = Vec::new();
        let mut max_align = 4usize;
        for (_, val) in fields {
            match val {
                Val::I64(_) => max_align = max_align.max(8),
                Val::I32(_) | Val::Offset(_) => max_align = max_align.max(4),
                Val::I16(_) => max_align = max_align.max(2),
                _ => {}
            }
        }

        for &(id, ref val) in fields {
            match *val {
                Val::U8(v) | Val::UnionType(v) => {
                    slot_off[id as usize] = (4 + inline.len()) as i16;
                    inline.push(v);
                }
                Val::Bool(v) => {
                    slot_off[id as usize] = (4 + inline.len()) as i16;
                    inline.push(u8::from(v));
                }
                Val::I16(v) => {
                    while (4 + inline.len()) % 2 != 0 {
                        inline.push(0);
                    }
                    slot_off[id as usize] = (4 + inline.len()) as i16;
                    inline.extend_from_slice(&v.to_le_bytes());
                }
                Val::I32(v) => {
                    while (4 + inline.len()) % 4 != 0 {
                        inline.push(0);
                    }
                    slot_off[id as usize] = (4 + inline.len()) as i16;
                    inline.extend_from_slice(&v.to_le_bytes());
                }
                Val::I64(v) => {
                    while (4 + inline.len()) % 8 != 0 {
                        inline.push(0);
                    }
                    slot_off[id as usize] = (4 + inline.len()) as i16;
                    inline.extend_from_slice(&v.to_le_bytes());
                }
                Val::Offset(target) => {
                    while (4 + inline.len()) % 4 != 0 {
                        inline.push(0);
                    }
                    slot_off[id as usize] = (4 + inline.len()) as i16;
                    offset_fixups.push((inline.len(), target));
                    inline.extend_from_slice(&0i32.to_le_bytes());
                }
            }
        }

        self.align(2);
        let vt_pos = self.buf.len();
        let vt_size = (4 + nslots * 2) as i16;
        let obj_size = (4 + inline.len()) as i16;
        self.buf.extend_from_slice(&vt_size.to_le_bytes());
        self.buf.extend_from_slice(&obj_size.to_le_bytes());
        for &s in &slot_off {
            self.buf.extend_from_slice(&s.to_le_bytes());
        }

        self.align(max_align);
        let obj_pos = self.buf.len();
        let soffset = obj_pos as i32 - vt_pos as i32;
        self.buf.extend_from_slice(&soffset.to_le_bytes());

        for (inline_idx, target) in offset_fixups {
            let slot_abs = obj_pos + 4 + inline_idx;
            let rel = target as i32 - slot_abs as i32;
            inline[inline_idx..inline_idx + 4].copy_from_slice(&rel.to_le_bytes());
        }

        self.buf.extend_from_slice(&inline);
        obj_pos as u32
    }

    pub fn finish(mut self, root: u32) -> Vec<u8> {
        self.buf[0..4].copy_from_slice(&(root as i32).to_le_bytes());
        self.buf
    }
}

impl Default for Fbb {
    fn default() -> Self {
        Self::new()
    }
}

pub fn root_as(buf: &[u8]) -> usize {
    let off = i32::from_le_bytes(buf[0..4].try_into().unwrap());
    off as usize
}

pub fn table_field_u8(buf: &[u8], table: usize, field_id: usize) -> Option<u8> {
    let slot = vtable_slot(buf, table, field_id)?;
    if slot == 0 {
        return None;
    }
    Some(buf[table + slot as usize])
}

pub fn table_field_bool(buf: &[u8], table: usize, field_id: usize) -> Option<bool> {
    table_field_u8(buf, table, field_id).map(|v| v != 0)
}

pub fn table_field_i16(buf: &[u8], table: usize, field_id: usize) -> Option<i16> {
    let slot = vtable_slot(buf, table, field_id)?;
    if slot == 0 {
        return None;
    }
    let p = table + slot as usize;
    Some(i16::from_le_bytes(buf[p..p + 2].try_into().unwrap()))
}

pub fn table_field_i32(buf: &[u8], table: usize, field_id: usize) -> Option<i32> {
    let slot = vtable_slot(buf, table, field_id)?;
    if slot == 0 {
        return None;
    }
    let p = table + slot as usize;
    Some(i32::from_le_bytes(buf[p..p + 4].try_into().unwrap()))
}

pub fn table_field_i64(buf: &[u8], table: usize, field_id: usize) -> Option<i64> {
    let slot = vtable_slot(buf, table, field_id)?;
    if slot == 0 {
        return None;
    }
    let p = table + slot as usize;
    Some(i64::from_le_bytes(buf[p..p + 8].try_into().unwrap()))
}

pub fn table_field_offset(buf: &[u8], table: usize, field_id: usize) -> Option<usize> {
    let slot = vtable_slot(buf, table, field_id)?;
    if slot == 0 {
        return None;
    }
    let p = table + slot as usize;
    let rel = i32::from_le_bytes(buf[p..p + 4].try_into().unwrap());
    Some((p as i32 + rel) as usize)
}

pub fn table_field_union_type(buf: &[u8], table: usize, field_id: usize) -> Option<u8> {
    table_field_u8(buf, table, field_id)
}

fn vtable_slot(buf: &[u8], table: usize, field_id: usize) -> Option<i16> {
    let soff = i32::from_le_bytes(buf[table..table + 4].try_into().unwrap());
    let vt = (table as i32 - soff) as usize;
    let vt_size = i16::from_le_bytes(buf[vt..vt + 2].try_into().unwrap()) as usize;
    let idx = 4 + field_id * 2;
    if idx + 2 > vt_size {
        return Some(0);
    }
    Some(i16::from_le_bytes(buf[vt + idx..vt + idx + 2].try_into().unwrap()))
}

pub fn read_string(buf: &[u8], pos: usize) -> &str {
    let len = i32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
    std::str::from_utf8(&buf[pos + 4..pos + 4 + len]).expect("utf8 string")
}

pub fn read_offset_vector(buf: &[u8], pos: usize) -> Vec<usize> {
    let n = i32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
    let base = pos + 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let slot = base + i * 4;
        let rel = i32::from_le_bytes(buf[slot..slot + 4].try_into().unwrap());
        out.push((slot as i32 + rel) as usize);
    }
    out
}

pub fn read_struct_vector_16(buf: &[u8], pos: usize) -> Vec<[u8; 16]> {
    let n = i32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(n);
    let mut p = pos + 4;
    for _ in 0..n {
        let mut e = [0u8; 16];
        e.copy_from_slice(&buf[p..p + 16]);
        out.push(e);
        p += 16;
    }
    out
}
