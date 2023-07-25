use anyhow::Result;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::mem::size_of;
use std::slice;

const MAX_SLICE_LEN: usize = 1024 * 1024;
pub struct FloatReader<'a> {
    reader: &'a mut BufReader<File>,
}

impl<'a> FloatReader<'a> {
    pub fn new(reader: &'a mut BufReader<File>) -> Self {
        Self { reader }
    }

    pub fn read_vec(&mut self, len: usize) -> Result<Vec<f32>> {
        let f32_s = size_of::<f32>();
        let mut vec: Vec<f32> = Vec::with_capacity(len);
        let mut buffer = vec![0; len.min(MAX_SLICE_LEN) * f32_s];
        let mut offset = 0;
        while offset < len {
            let chunk_len = MAX_SLICE_LEN.min(len - offset);
            self.reader.read_exact(&mut buffer[0..chunk_len * f32_s])?;
            unsafe {
                let x = slice::from_raw_parts::<f32>(buffer.as_ptr() as *const f32, chunk_len)
                    .to_owned();
                vec.extend(&x);
            };
            offset += MAX_SLICE_LEN;
        }
        Ok(vec)
    }

    pub fn read_int(&mut self) -> Result<u32> {
        let mut buf = vec![0; size_of::<u32>()];
        self.reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]))
    }
}
