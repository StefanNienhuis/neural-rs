use std::io::{Cursor, Read};
use byteorder::{BigEndian, ReadBytesExt};

pub struct IDXImageFile {
    pub image_count: u32,
    pub row_count: u32,
    pub column_count: u32,
    pub images: Vec<Vec<Vec<u8>>>
}

pub fn parse_idx_image_file(data: Vec<u8>) -> Result<IDXImageFile, String> {
    let mut cursor = Cursor::new(data);

    // Skip magic number
    let _ = cursor.read_u32::<BigEndian>();

    let image_count = cursor.read_u32::<BigEndian>().or_else(|error| Err(format!("Error while decoding IDX: {}", error)))?;
    let row_count = cursor.read_u32::<BigEndian>().or_else(|error| Err(format!("Error while decoding IDX: {}", error)))?;
    let column_count = cursor.read_u32::<BigEndian>().or_else(|error| Err(format!("Error while decoding IDX: {}", error)))?;

    let mut pixels = Vec::new();

    cursor.read_to_end(&mut pixels).or_else(|error| Err(format!("Error while decoding IDX: {}", error)))?;

    let rows: Vec<Vec<u8>> = pixels.chunks(column_count as usize).map(|x| x.to_vec()).collect();
    let images: Vec<Vec<Vec<u8>>> = rows.chunks(row_count as usize).map(|x| x.to_vec()).collect();

    if images.len() != image_count as usize {
        return Err(format!("Error while decoding IDX: Image count ({}) is not equal to parsed image count ({})", image_count, images.len()))
    }

    return Ok(IDXImageFile {
        image_count, row_count, column_count, images
    });
}

pub struct IDXLabelFile {
    pub item_count: u32,
    pub labels: Vec<u8>
}

pub fn parse_idx_label_file(data: Vec<u8>) -> Result<IDXLabelFile, String> {
    let mut cursor = Cursor::new(data);

    // Skip magic number
    let _ = cursor.read_u32::<BigEndian>();

    let item_count = cursor.read_u32::<BigEndian>().or_else(|error| Err(format!("Error while decoding IDX: {}", error)))?;

    let mut labels = Vec::new();

    cursor.read_to_end(&mut labels).or_else(|error| Err(format!("Error while decoding IDX: {}", error)))?;

    if labels.len() != item_count as usize {
        return Err(format!("Error while decoding IDX: Item count ({}) is not equal to parsed label count ({})", item_count, labels.len()))
    }

    return Ok(IDXLabelFile {
        item_count, labels
    });
}