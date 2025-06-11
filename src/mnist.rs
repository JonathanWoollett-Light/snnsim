use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use reqwest::blocking;
use std::io::{Cursor, Read};
use thiserror::Error;

const BASE_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/";

pub fn get_training() -> Result<(Vec<Vec<u8>>, Vec<u8>), GetMnistDataError> {
    get_mnist_data("train")
}
pub fn get_testing() -> Result<(Vec<Vec<u8>>, Vec<u8>), GetMnistDataError> {
    get_mnist_data("t10k")
}

#[derive(Debug,Error)]
pub enum GetMnistDataError {
    #[error("Failed to donwload images: {0}")]
    DownloadImages(DownloadAndDecompressError),
    #[error("Failed to parse images: {0}")]
    ParseImages(Box<dyn std::error::Error>),
    #[error("Failed to donwload labels: {0}")]
    DownloadLabels(DownloadAndDecompressError),
    #[error("Failed to parse labels: {0}")]
    ParseLabels(Box<dyn std::error::Error>)
}

fn get_mnist_data(prefix: &str) -> Result<(Vec<Vec<u8>>, Vec<u8>), GetMnistDataError> {
    // Download image data
    let img_data = download_and_decompress(&format!("{}-images-idx3-ubyte.gz", prefix)).map_err(GetMnistDataError::DownloadImages)?;
    let images = parse_idx_images(&img_data).map_err(GetMnistDataError::ParseImages)?;

    // Download label data
    let label_data = download_and_decompress(&format!("{}-labels-idx1-ubyte.gz", prefix)).map_err(GetMnistDataError::DownloadLabels)?;
    let labels = parse_idx_labels(&label_data).map_err(GetMnistDataError::ParseLabels)?;

    Ok((images, labels))
}

#[derive(Debug,Error)]
pub enum DownloadAndDecompressError {
    #[error("Failed to get response: {0}")]
    Get(reqwest::Error),
    #[error("Failed to get bytes: {0}")]
    Bytes(reqwest::Error),
    #[error("Failed to read bytes: {0}")]
    Read(std::io::Error)
}

fn download_and_decompress(filename: &str) -> Result<Vec<u8>, DownloadAndDecompressError> {
    let url = format!("{}{}", BASE_URL, filename);
    let response = blocking::get(&url).map_err(DownloadAndDecompressError::Get)?.bytes().map_err(DownloadAndDecompressError::Bytes)?;
    println!("downloaded");
    let mut decompressed = Vec::new();
    GzDecoder::new(&response[..]).read_to_end(&mut decompressed).map_err(DownloadAndDecompressError::Read)?;
    println!("decompressed");
    Ok(decompressed)
}

fn parse_idx_images(data: &[u8]) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let mut cursor = Cursor::new(data);

    if cursor.read_i32::<BigEndian>()? != 2051 {
        return Err("Invalid image file magic number".into());
    }

    let count = cursor.read_i32::<BigEndian>()? as usize;
    let rows = cursor.read_i32::<BigEndian>()? as usize;
    let cols = cursor.read_i32::<BigEndian>()? as usize;
    let image_size = rows * cols;

    let mut images = Vec::with_capacity(count);
    for _ in 0..count {
        let mut image = vec![0u8; image_size];
        cursor.read_exact(&mut image)?;
        images.push(image);
    }

    Ok(images)
}

fn parse_idx_labels(data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut cursor = Cursor::new(data);
    if cursor.read_i32::<BigEndian>()? != 2049 {
        return Err("Invalid label file magic number".into());
    }

    let count = cursor.read_i32::<BigEndian>()? as usize;
    let mut labels = vec![0u8; count];
    cursor.read_exact(&mut labels)?;

    Ok(labels)
}
