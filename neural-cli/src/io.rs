use std::fs;
use crate::{idx, idx::{IDXImageFile, IDXLabelFile}};

use std::{path::PathBuf};
use bincode;
use neural::network::Network;

pub fn read_file(path: &PathBuf) -> Result<Vec::<u8>, String> {
    if !path.exists() {
        return Err(format!("File does not exist"));
    }

    match fs::read(path) {
        Err(error) => Err(format!("Couldn't read file: {}", error)),
        Ok(data) => Ok(data)
    }
}

pub fn write_file(data: &Vec<u8>, path: &PathBuf, new: bool, extension: Option<&str>) -> Result<(), String> {
    if new && path.exists() {
        return Err(format!("File already exists"));
    } else if !new && !path.exists() {
        return Err(format!("File does not exist"));
    } else if let Some(extension) = extension {
        if path.extension().is_none() || path.extension().unwrap() != extension {
            return Err(format!("File should end with '{}' extension", extension))
        }
    }

    match fs::write(path, data.as_slice()) {
        Err(error) => {
            Err(format!("{}", error))
        },
        Ok(_) => Ok(())
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