//! Weights: mmap a safetensors checkpoint and expose tensors as `&[f32]`.
//!
//! The expected file is a raw safetensors checkpoint (`model.safetensors`),
//! as produced by `safetensors.torch.save_file`: an 8-byte little-endian
//! header length, that many bytes of JSON, then the tensor payloads. Header
//! keys carry `dtype`, `shape` and `data_offsets` (relative to the end of the
//! header). F32 tensors are read straight out of the mapping; I64 tensors
//! (the `relative_position_index` buffers) are converted to f32 on first use.
//!
//! `get()` returns a zero-copy `&[f32]` into the mmap for F32 data.
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::io::AsRawFd;

/// Tensor dtypes understood in a raw safetensors header. `Int` buffers are
/// materialised as f32 on first access (all of them are small exact integers);
/// `F32` tensors are viewed in place.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    F32,
    Int,
}

pub struct Blob {
    ptr: *const u8,
    len: usize,
}

impl Blob {
    pub fn open(path: &str) -> Result<Blob, String> {
        let f = File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
        let len = f.metadata().map_err(|e| e.to_string())?.len() as usize;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                f.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(format!("mmap {} failed", path));
        }
        Ok(Blob { ptr: ptr as *const u8, len })
    }

    #[inline]
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub offset: usize,
    pub nbytes: usize,
    /// Element size of the payload in the file (4 for f32, 8 for i64).
    /// Conversion to f32 is decided from this.
    pub esize: usize,
}

impl TensorInfo {
    pub fn is_int(&self) -> bool {
        self.esize != 4
    }
}

pub struct Weights {
    pub blob: Blob,
    pub tensors: HashMap<String, TensorInfo>,
    pub total_bytes: usize,
    /// Tensors that had to be materialised rather than viewed in place:
    /// I64 (and I32) buffers converted to f32, keyed by tensor name.
    owned: RefCell<HashMap<String, Vec<f32>>>,
}

impl Weights {
    /// Mmap a raw safetensors checkpoint and build the tensor index from the
    /// header embedded in the file itself.
    pub fn open(path: &str) -> Result<Weights, String> {
        let blob = Blob::open(path)?;
        let mut w = Weights {
            blob,
            tensors: HashMap::new(),
            total_bytes: 0,
            owned: RefCell::new(HashMap::new()),
        };
        let (tensors, total) = parse_safetensors_index(w.blob.bytes())?;
        w.tensors = tensors;
        w.total_bytes = total;
        Ok(w)
    }

    pub fn get(&self, name: &str) -> Result<&[f32], String> {
        if self.dtype(name)? == Dtype::Int {
            return self.converted(name);
        }
        self.get_raw(name)
    }

    /// Dtype recorded for a tensor in the safetensors header.
    pub fn dtype(&self, name: &str) -> Result<Dtype, String> {
        match self.tensors.get(name) {
            Some(t) if t.is_int() => Ok(Dtype::Int),
            Some(_) => Ok(Dtype::F32),
            None => Err(format!("missing tensor {}", name)),
        }
    }

    /// Integer accessor for the I64 `relative_position_index` buffers. Only
    /// meaningful for tensors whose `esize` is 8.
    pub fn get_i64(&self, name: &str) -> Result<Vec<i64>, String> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {}", name))?;
        if t.esize != 8 {
            return Err(format!("tensor {} is not an i64 tensor", name));
        }
        let all = self.blob.bytes();
        let end = t.offset.checked_add(t.nbytes).ok_or("offset overflow")?;
        if end > all.len() {
            return Err(format!("tensor {} out of blob bounds", name));
        }
        if t.offset % 8 != 0 {
            return Err(format!("tensor {} not 8-byte aligned", name));
        }
        let p = unsafe {
            std::slice::from_raw_parts(all.as_ptr().add(t.offset) as *const i64, t.nbytes / 8)
        };
        Ok(p.to_vec())
    }

    fn get_raw(&self, name: &str) -> Result<&[f32], String> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {}", name))?;
        let end = t.offset + t.nbytes;
        let all = self.blob.bytes();
        if end > all.len() {
            return Err(format!("tensor {} out of blob bounds", name));
        }
        if t.nbytes % 4 != 0 || t.offset % 4 != 0 {
            return Err(format!("tensor {} not 4-byte aligned", name));
        }
        let s = &all[t.offset..end];
        Ok(unsafe { std::slice::from_raw_parts(s.as_ptr() as *const f32, t.nbytes / 4) })
    }

    /// View a materialised (converted) tensor, e.g. an I64 buffer read as f32.
    fn converted(&self, name: &str) -> Result<&[f32], String> {
        if self.owned.borrow().contains_key(name) {
            let c = self.owned.borrow();
            let v: &Vec<f32> = c.get(name).unwrap();
            let p: *const f32 = v.as_ptr();
            drop(c);
            return Ok(unsafe { std::slice::from_raw_parts(p, self.tensors[name].shape.iter().product()) });
        }
        let vals = self.get_i64(name)?;
        let f: Vec<f32> = vals.iter().map(|v| *v as f32).collect();
        self.owned.borrow_mut().insert(name.to_string(), f);
        self.converted(name)
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], String> {
        Ok(&self.tensors.get(name).ok_or_else(|| format!("missing tensor {}", name))?.shape)
    }

    /// Fetch a tensor and assert its shape (in the checkpoint's own layout).
    pub fn get_shaped(&self, name: &str, shape: &[usize]) -> Result<&[f32], String> {
        let t = self.tensors.get(name).ok_or_else(|| format!("missing tensor {}", name))?;
        if t.shape != shape {
            return Err(format!("tensor {} shape {:?} != expected {:?}", name, t.shape, shape));
        }
        self.get(name)
    }

    #[allow(dead_code)] // useful for debugging/verification tools
    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();
        v.sort_unstable();
        v
    }
}

/// Parse a safetensors header into a `name -> TensorInfo` map.
/// `data_offsets` are relative to the end of the header, so the stored
/// offsets are absolute positions in the mmapped file.
///
/// Tensors whose payload is not plain f32 (the I64 `relative_position_index`
/// buffers, and any I32 scalar like `num_batches_tracked`) carry their true
/// element size in `TensorInfo::esize` and are converted to f32 on first
/// access; everything else is viewed in place. Returns the map plus the number
/// of payload bytes after the header.
fn parse_safetensors_index(blob: &[u8]) -> Result<(HashMap<String, TensorInfo>, usize), String> {
    if blob.len() < 8 {
        return Err("safetensors: file too small".to_string());
    }
    let hlen = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
    if hlen == 0 || hlen > 1 << 34 {
        return Err(format!("safetensors: implausible header length {}", hlen));
    }
    let start = 8usize;
    let end = start
        .checked_add(hlen)
        .filter(|e| *e <= blob.len())
        .ok_or_else(|| "safetensors: truncated header".to_string())?;
    let head = std::str::from_utf8(&blob[start..end])
        .map_err(|_| "safetensors: header is not utf-8".to_string())?;
    if !head.starts_with('{') {
        return Err("safetensors: header does not start with '{'".to_string());
    }
    let mut tensors = HashMap::new();
    let mut search = 0usize;
    while let Some(p) = find_from(head, "\"dtype\"", search) {
        // Tensor objects always start with `"dtype"`, so the header text just
        // before it is `"<name>":{"dtype"`. `brace` is the object's own `{`,
        // `brace - 1` the `:`, `brace - 2` the closing quote of the name key.
        let brace = head[..p].rfind('{').ok_or("safetensors: no object before dtype")?;
        if brace < 3 {
            return Err("safetensors: dtype outside any object".to_string());
        }
        let key_end = brace - 2;
        if head.as_bytes()[key_end] != b'"' {
            return Err("safetensors: malformed key before dtype".to_string());
        }
        let key_start = head[..key_end]
            .rfind('"')
            .ok_or("safetensors: no name key before dtype")?;
        let name = head[key_start + 1..key_end].to_string();
        if name != "__metadata__" {
            let dtype = parse_string_after(head, "\"dtype\"", p)
                .ok_or("safetensors: missing dtype value")?;
            let p_shape = find_from(head, "\"shape\"", p).ok_or("safetensors: missing shape")?;
            let shape = parse_shape_after(head, p_shape).ok_or("safetensors: bad shape")?;
            let p_off = find_from(head, "\"data_offsets\"", p_shape)
                .ok_or("safetensors: missing data_offsets")?;
            let (a, b) = parse_usize_pair_after(head, "\"data_offsets\"", p_off)
                .ok_or("safetensors: bad data_offsets")?;
            if b < a || end + b > blob.len() {
                return Err(format!("safetensors: tensor {} out of bounds", name));
            }
            let count: usize = shape.iter().product();
            let nbytes = b - a;
            let esize = match dtype.as_str() {
                "F32" => 4usize,
                "I64" => 8,
                "I32" => 4,
                other => {
                    return Err(format!("safetensors: unsupported dtype {} for {}", other, name))
                }
            };
            if count * esize != nbytes {
                return Err(format!(
                    "safetensors: tensor {} has {} bytes for {:?} ({} element(s) of {})",
                    name, nbytes, shape, count, dtype
                ));
            }
            let stored = end + a;
            tensors.insert(
                name,
                TensorInfo { shape, offset: stored, nbytes, esize },
            );
            search = p + 8;
        } else {
            // Skip metadata by jumping past its dtype token.
            search = p + 8;
        }
    }
    if tensors.is_empty() {
        return Err("safetensors: no tensors found in header".to_string());
    }
    // The payload length is the header length plus the largest end offset, so
    // recall it from the raw JSON rather than from the converted entries.
    let mut tail = 0usize;
    let mut s = 0usize;
    while let Some(p) = find_from(head, "\"data_offsets\"", s) {
        if let Some((_, b)) = parse_usize_pair_after(head, "\"data_offsets\"", p) {
            tail = tail.max(b);
        }
        s = p + 14;
    }
    if end + tail > blob.len() {
        return Err("safetensors: payload extends past end of file".to_string());
    }
    Ok((tensors, blob.len() - 8 - hlen))
}

fn find_from(hay: &str, needle: &str, from: usize) -> Option<usize> {
    if from >= hay.len() {
        return None;
    }
    hay[from..].find(needle).map(|p| p + from)
}


/// Read the two integers of a JSON array such as `"data_offsets":[a,b]`,
/// starting at the `key` token itself.
fn parse_usize_pair_after(hay: &str, key: &str, pos: usize) -> Option<(usize, usize)> {
    let p = find_from(hay, key, pos)?;
    let mut rest = &hay[p + key.len()..];
    fn skip(s: &str) -> &str {
        s.trim_start_matches(|c: char| c.is_whitespace() || c == ':' || c == '[' || c == ',')
    }
    rest = skip(rest);
    let e = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    let a = rest[..e].parse::<usize>().ok()?;
    rest = skip(&rest[e..]);
    let e = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    let b = rest[..e].parse::<usize>().ok()?;
    Some((a, b))
}

fn parse_string_after(hay: &str, key: &str, pos: usize) -> Option<String> {
    let p = find_from(hay, key, pos)?;
    let rest = &hay[p + key.len()..];
    let start = rest.find('"')? + 1;
    let end = rest[start..].find('"')? + start;
    Some(rest[start..end].to_string())
}

fn parse_shape_after(hay: &str, pos: usize) -> Option<Vec<usize>> {
    let open = hay[pos..].find('[')? + pos;
    let close = hay[open..].find(']')? + open;
    let inner = &hay[open + 1..close];
    if inner.trim().is_empty() {
        return Some(vec![]);
    }
    inner
        .split(',')
        .map(|s| s.trim().parse::<usize>().ok())
        .collect::<Option<Vec<usize>>>()
}
