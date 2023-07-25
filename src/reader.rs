use anyhow::Result;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::mem::size_of;
use std::slice;

pub struct FloatReader<'a> {
    reader: &'a mut BufReader<File>,
}

impl<'a> FloatReader<'a> {
    pub fn new(reader: &'a mut BufReader<File>) -> Self {
        Self { reader }
    }

    pub fn read_vec(&mut self, len: i32) -> Result<Vec<f32>> {
        let f32_s = size_of::<f32>() as usize;
        let len = len as usize;
        // TODO(poudels14): use Buffer
        let mut buffer = vec![0; len * f32_s];
        self.reader.read_exact(&mut buffer).unwrap();
        unsafe {
            let x = slice::from_raw_parts::<f32>(buffer.as_ptr() as *const f32, len).to_owned();
            return Ok(x);
        };
    }
}
