use std::io::{Cursor, Read};
use byteorder::{BigEndian, ReadBytesExt};

pub struct IDXFile {
    pub shape: Vec<u32>,
    pub items: Vec<Vec<u8>>
}

pub fn parse_idx_file(data: Vec<u8>) -> Result<IDXFile, String> {
    let mut cursor = Cursor::new(data);

    // Skip magic number
    let _ = cursor.read_u16::<BigEndian>();

    let data_type = cursor.read_u8().or_else(|error| Err(format!("Error while decoding IDX: {}", error)))?;

    assert_eq!(data_type, 0x08, "Only IDX files with unsigned byte data types are supported.");

    let dimensions = cursor.read_u8().or_else(|error| Err(format!("Error while decoding IDX: {}", error)))?;

    let mut shape = vec![];

    for _ in 0..dimensions {
        shape.push(cursor.read_u32::<BigEndian>().or_else(|error| Err(format!("Error while decoding IDX: {}", error)))?);
    }

    let mut items = Vec::new();

    cursor.read_to_end(&mut items).or_else(|error| Err(format!("Error while decoding IDX: {}", error)))?;

    let expected_count = shape.iter().fold(1, |a,x| a * x) as usize;

    if items.len() != expected_count {
        return Err(format!("Error while decoding IDX: Expected item count ({}) is not equal to parsed item count ({})", expected_count, items.len()))
    }

    return Ok(IDXFile {
        items: items.chunks(items.len() / shape[0].clone() as usize).map(|x| x.to_vec()).collect(), shape
    });
}