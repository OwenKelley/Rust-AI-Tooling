//! Reverse FlatBuffers builder (Google / Arrow C++ style).
//! Writes high→low so nested uoffsets are always positive for PyArrow's verifier.
//!
//! Offsets returned by `create_*` are distances from the **buffer end** (stable
//! under reallocation). `finish` rewrites the root into a compact buffer.

#[derive(Clone, Debug)]
pub enum Val {
    U8(u8),
    Bool(bool),
    I16(i16),
    I32(i32),
    I64(i64),
    /// Distance from buffer end to the target object.
    Offset(u32),
    UnionType(u8),
}

pub struct RevFbb {
    buf: Vec<u8>,
    head: usize,
    minalign: usize,
    /// Per-field positions as distance from buffer end (`0` = unset).
    cur_vt_from_end: Vec<usize>,
    object_end_from_end: usize,
    nested: bool,
}

impl RevFbb {
    pub fn new() -> Self {
        let cap = 64 * 1024;
        Self {
            buf: vec![0u8; cap],
            head: cap,
            minalign: 1,
            cur_vt_from_end: Vec::new(),
            object_end_from_end: 0,
            nested: false,
        }
    }

    fn from_end(&self, abs: usize) -> u32 {
        (self.buf.len() - abs) as u32
    }

    fn to_abs(&self, from_end: u32) -> usize {
        self.buf.len() - from_end as usize
    }

    fn ensure(&mut self, need: usize) {
        if self.head >= need {
            return;
        }
        let used = self.buf.len() - self.head;
        let new_len = (self.buf.len() * 2 + need + used).next_power_of_two();
        let mut nb = vec![0u8; new_len];
        let new_head = new_len - used;
        nb[new_head..].copy_from_slice(&self.buf[self.head..]);
        self.buf = nb;
        self.head = new_head;
    }

    fn prep(&mut self, align: usize, size: usize) {
        self.minalign = self.minalign.max(align);
        while (self.buf.len() - self.head + size) % align != 0 {
            self.ensure(1);
            self.head -= 1;
            self.buf[self.head] = 0;
        }
    }

    fn push_bytes(&mut self, bytes: &[u8]) {
        self.ensure(bytes.len());
        self.head -= bytes.len();
        self.buf[self.head..self.head + bytes.len()].copy_from_slice(bytes);
    }

    fn push_u8(&mut self, v: u8) {
        self.prep(1, 1);
        self.push_bytes(&[v]);
    }

    fn push_i16(&mut self, v: i16) {
        self.prep(2, 2);
        self.push_bytes(&v.to_le_bytes());
    }

    fn push_i32(&mut self, v: i32) {
        self.prep(4, 4);
        self.push_bytes(&v.to_le_bytes());
    }

    fn push_i64(&mut self, v: i64) {
        self.prep(8, 8);
        self.push_bytes(&v.to_le_bytes());
    }

    pub fn create_string_off(&mut self, s: &str) -> u32 {
        assert!(!self.nested);
        self.prep(4, s.len() + 1 + 4);
        let mut tmp = Vec::with_capacity(s.len() + 1);
        tmp.extend_from_slice(s.as_bytes());
        tmp.push(0);
        self.push_bytes(&tmp);
        self.push_i32(s.len() as i32);
        self.from_end(self.head)
    }

    pub fn create_offset_vector(&mut self, offs: &[u32]) -> u32 {
        assert!(!self.nested);
        self.prep(4, 4 + offs.len() * 4);
        for _ in offs.iter().rev() {
            self.push_i32(0);
        }
        let slots_start = self.head;
        self.push_i32(offs.len() as i32);
        let vec_start = self.head;
        for (i, &o_fe) in offs.iter().enumerate() {
            let slot = slots_start + i * 4;
            let target = self.to_abs(o_fe);
            let rel = target as i32 - slot as i32;
            assert!(rel > 0, "uoffset must be forward: {rel}");
            self.buf[slot..slot + 4].copy_from_slice(&rel.to_le_bytes());
        }
        self.from_end(vec_start)
    }

    pub fn create_struct_vector_16(&mut self, elems: &[[u8; 16]]) -> u32 {
        assert!(!self.nested);
        self.prep(8, elems.len() * 16);
        for e in elems.iter().rev() {
            self.push_bytes(e);
        }
        self.push_i32(elems.len() as i32);
        self.from_end(self.head)
    }

    pub fn create_struct_vector_24(&mut self, elems: &[[u8; 24]]) -> u32 {
        assert!(!self.nested);
        self.prep(8, elems.len() * 24);
        for e in elems.iter().rev() {
            self.push_bytes(e);
        }
        self.push_i32(elems.len() as i32);
        self.from_end(self.head)
    }

    fn start_table(&mut self, num_fields: usize) {
        assert!(!self.nested);
        self.nested = true;
        self.cur_vt_from_end = vec![0; num_fields];
        self.object_end_from_end = self.from_end(self.head) as usize;
    }

    fn add_slot(&mut self, id: usize, write: impl FnOnce(&mut Self)) {
        if self.cur_vt_from_end[id] != 0 {
            return;
        }
        write(self);
        self.cur_vt_from_end[id] = self.from_end(self.head) as usize;
    }

    fn end_table(&mut self) -> u32 {
        assert!(self.nested);
        self.prep(4, 4);
        self.push_i32(0);
        let obj = self.head;
        let object_end = self.to_abs(self.object_end_from_end as u32);
        let obj_size = (object_end - obj) as i16;

        let nslots = self.cur_vt_from_end.len();
        let vt_size = (4 + nslots * 2) as i16;
        let mut vt = Vec::with_capacity(4 + nslots * 2);
        vt.extend_from_slice(&vt_size.to_le_bytes());
        vt.extend_from_slice(&obj_size.to_le_bytes());
        for &fe in &self.cur_vt_from_end {
            let slot = if fe == 0 {
                0i16
            } else {
                let abs = self.to_abs(fe as u32);
                (abs - obj) as i16
            };
            vt.extend_from_slice(&slot.to_le_bytes());
        }
        self.prep(2, vt.len());
        self.push_bytes(&vt);
        let vt_pos = self.head;

        let soff = (obj as i32) - (vt_pos as i32);
        self.buf[obj..obj + 4].copy_from_slice(&soff.to_le_bytes());

        self.nested = false;
        self.cur_vt_from_end.clear();
        self.object_end_from_end = 0;
        self.from_end(obj)
    }

    /// `fields` are added in **slice order** (first = Add first = highest address).
    pub fn create_table(&mut self, fields: &[(u16, Val)]) -> u32 {
        let nslots = if fields.is_empty() {
            0
        } else {
            fields.iter().map(|(id, _)| *id).max().unwrap() as usize + 1
        };
        self.start_table(nslots);
        for &(id, ref val) in fields {
            let id = id as usize;
            match *val {
                Val::U8(v) | Val::UnionType(v) => {
                    self.add_slot(id, |b| b.push_u8(v));
                }
                Val::Bool(v) => {
                    self.add_slot(id, |b| b.push_u8(u8::from(v)));
                }
                Val::I16(v) => {
                    self.add_slot(id, |b| b.push_i16(v));
                }
                Val::I32(v) => {
                    self.add_slot(id, |b| b.push_i32(v));
                }
                Val::I64(v) => {
                    self.add_slot(id, |b| b.push_i64(v));
                }
                Val::Offset(target_fe) => {
                    self.add_slot(id, |b| {
                        b.prep(4, 4);
                        let slot = b.head - 4;
                        let target = b.to_abs(target_fe);
                        let rel = target as i32 - slot as i32;
                        assert!(rel > 0, "uoffset must be forward: {rel}");
                        b.push_bytes(&rel.to_le_bytes());
                    });
                }
            }
        }
        self.end_table()
    }

    pub fn finish(mut self, root_fe: u32) -> Vec<u8> {
        assert!(!self.nested);
        let root_abs = self.to_abs(root_fe);
        self.prep(self.minalign.max(4), 4);
        let root_loc = self.head - 4;
        let rel = root_abs as i32 - root_loc as i32;
        assert!(rel > 0, "root uoffset must be forward");
        self.push_i32(rel);
        // Compact: relatives inside are unchanged when slicing from head.
        self.buf[self.head..].to_vec()
    }
}

impl Default for RevFbb {
    fn default() -> Self {
        Self::new()
    }
}
