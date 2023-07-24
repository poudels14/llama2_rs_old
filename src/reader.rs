use anyhow::Result;
use nalgebra::DVector;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::mem::size_of;
use std::slice;

pub struct ModelLoader<'a> {
    reader: &'a mut BufReader<File>,
}

impl<'a> ModelLoader<'a> {
    pub fn new(reader: &'a mut BufReader<File>) -> Self {
        Self { reader }
    }

    /// Returns (m * n) "matrix"
    pub fn read_matrix(&mut self, m: usize, n: usize) -> Result<Vec<DVector<f32>>> {
        let f32_s = size_of::<f32>() as usize;
        let len = (m * n * f32_s) as usize;
        // TODO(poudels14): use Buffer
        let mut buffer = vec![0; len];
        self.reader.read_exact(&mut buffer)?;

        unsafe {
            let x = slice::from_raw_parts::<f32>(buffer.as_ptr() as *const f32, m * n);
            Ok(x.chunks_exact(n as usize)
                .map(|c| DVector::from_row_slice(c))
                .collect())
        }
    }

    pub fn read_vec(&mut self, len: usize) -> Result<Vec<f32>> {
        let f32_s = size_of::<f32>() as usize;
        let len = len as usize;
        // TODO(poudels14): use Buffer
        let mut buffer = vec![0; len * f32_s];
        self.reader.read_exact(&mut buffer)?;
        unsafe {
            let x = slice::from_raw_parts::<f32>(buffer.as_ptr() as *const f32, len).to_owned();
            return Ok(x);
        };
    }

    pub fn read_int(&mut self) -> Result<i32> {
        let mut buffer = vec![0; size_of::<i32>()];
        self.reader.read_exact(&mut buffer)?;
        Ok(bincode::deserialize(&buffer)?)
    }
}
