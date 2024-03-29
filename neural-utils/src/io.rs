use crate::{
    idx::{self, IDXFile}
};
use neural::Network;
use std::fs;

use bincode;
use std::path::PathBuf;

pub fn read_file(path: &PathBuf) -> Result<Vec<u8>, String> {
    if !path.exists() {
        return Err(format!("File does not exist"));
    }

    match fs::read(path) {
        Err(error) => Err(format!("Couldn't read file: {}", error)),
        Ok(data) => Ok(data),
    }
}

pub fn write_file(
    data: &Vec<u8>,
    path: &PathBuf,
    new: bool,
    extension: Option<&str>,
) -> Result<(), String> {
    if new && path.exists() {
        return Err(format!("File already exists"));
    } else if !new && !path.exists() {
        return Err(format!("File does not exist"));
    } else if let Some(extension) = extension {
        if path.extension().is_none() || path.extension().unwrap() != extension {
            return Err(format!("File should end with '{}' extension", extension));
        }
    }

    match fs::write(path, data.as_slice()) {
        Err(error) => Err(format!("{}", error)),
        Ok(_) => Ok(()),
    }
}

pub fn read_network_file(path: &PathBuf) -> Result<Network, String> {
    let data = match read_file(path) {
        Err(error) => return Err(error),
        Ok(data) => data,
    };

    match bincode::decode_from_slice(data.as_slice(), bincode::config::standard()) {
        Err(error) => Err(format!("Error while parsing network: {}", error)),
        Ok((network, _)) => Ok(network),
    }
}

pub fn read_idx_file(path: &PathBuf) -> Result<IDXFile, String> {
    let data = match read_file(path) {
        Err(error) => return Err(error),
        Ok(data) => data,
    };

    match idx::parse_idx_file(data) {
        Err(error) => Err(error),
        Ok(idx) => Ok(idx),
    }
}
