use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use reqwest::blocking;
use std::io::{Cursor, Read};

const BASE_URL: &str = "https://github.com/cvdfoundation/mnist/raw/master/";

pub fn get_training() -> Result<(Vec<Vec<u8>>, Vec<u8>), Box<dyn std::error::Error>> {
    get_mnist_data("train")
}
pub fn get_testing() -> Result<(Vec<Vec<u8>>, Vec<u8>), Box<dyn std::error::Error>> {
    get_mnist_data("t10k")
}

fn get_mnist_data(prefix: &str) -> Result<(Vec<Vec<u8>>, Vec<u8>), Box<dyn std::error::Error>> {
    // Download image data
    let img_data = download_and_decompress(&format!("{}-images-idx3-ubyte.gz", prefix))?;
    let images = parse_idx_images(&img_data)?;

    // Download label data
    let label_data = download_and_decompress(&format!("{}-labels-idx1-ubyte.gz", prefix))?;
    let labels = parse_idx_labels(&label_data)?;

    Ok((images, labels))
}

fn download_and_decompress(filename: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let url = format!("{}{}", BASE_URL, filename);
    let response = blocking::get(&url)?.bytes()?;
    let mut decompressed = Vec::new();
    GzDecoder::new(&response[..]).read_to_end(&mut decompressed)?;
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
