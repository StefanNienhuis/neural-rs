use crate::{Network, idx};

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use bincode;
use crate::idx::{IDXImageFile, IDXLabelFile};

pub fn read_file(path: &PathBuf) -> Result<Vec::<u8>, String> {
    if !path.exists() {
        return Err(format!("{} does not exist", path.display()));
    }

    let mut file = match File::open(path) {
        Err(error) => return Err(format!("Couldn't open {}: {}", path.display(), error)),
        Ok(file) => file
    };

    let mut data = Vec::<u8>::new();

    match file.read_to_end(&mut data) {
        Err(error) => Err(format!("Couldn't read {}: {}", path.display(), error)),
        Ok(_) => Ok(data)
    }
}

pub fn parse_network_file(path: &PathBuf) -> Result<Network, String> {
    let data = match read_file(path) {
        Err(error) => return Err(error),
        Ok(data) => data
    };

    match bincode::decode_from_slice(data.as_slice(), bincode::config::standard()) {
        Err(error) => Err(format!("Error while parsing network: {}", error)),
        Ok((network, _)) => Ok(network)
    }
}

pub fn parse_idx_image_file(path: &PathBuf) -> Result<IDXImageFile, String> {
    let data = match read_file(path) {
        Err(error) => return Err(error),
        Ok(data) => data
    };

    match idx::parse_idx_image_file(data) {
        Err(error) => Err(error),
        Ok(image_file) => Ok(image_file)
    }
}

pub fn parse_idx_label_file(path: &PathBuf) -> Result<IDXLabelFile, String> {
    let data = match read_file(path) {
        Err(error) => return Err(error),
        Ok(data) => data
    };

    match idx::parse_idx_label_file(data) {
        Err(error) => Err(error),
        Ok(label_file) => Ok(label_file)
    }
}