use byteorder::{BigEndian, ReadBytesExt};
use core::f32;
use flate2::read::GzDecoder;
use reqwest::blocking;
use std::io::{Cursor, Read};
use std::{
    cmp::Ordering,
    io::Write,
    sync::Arc,
    time::{Duration, Instant},
};
use thiserror::Error;

use cudarc::{
    cublas::{Asum, AsumConfig},
    driver::CudaStream,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::Axis;
use ndarray_rand::rand_distr;
use snnsim::{PollingIterator, cuda::CudaMatrix, encode::rate_coding};

const MNIST_IMAGE_SIZE: usize = 28 * 28;
const MNIST_LABELS: usize = 10;
const TIME_STEPS: usize = 10;
const BATCH_SIZE: usize = 10_000;
const PIXEL_FIDELITY: f32 = 255f32;
const EPOCHS: usize = 1_000;
const LEARNING_RATE: f32 = 0.2f32;
// const TRAINING_SAMPLES: usize = 60_000;
// const TESTING_SAMPLES: usize = 10_000;
// const TRAIN_BATCHES: usize = TRAINING_SAMPLES / BATCH_SIZE;
// const TEST_BATCHES: usize = TESTING_SAMPLES / BATCH_SIZE;
const TMP: &str = env!("CARGO_TARGET_TMPDIR");

fn transform_data(
    (images, labels): (Vec<Vec<u8>>, Vec<u8>),
    stream: Arc<CudaStream>,
) -> (Vec<Vec<CudaMatrix>>, Vec<Vec<CudaMatrix>>) {
    let n = images.len();
    assert_eq!(n, labels.len());
    assert_eq!(n % BATCH_SIZE, 0);

    // Row-major
    let images = images
        .into_iter()
        .flatten()
        .map(|x| x as f32 / PIXEL_FIDELITY)
        .collect();
    let images = ndarray::Array2::from_shape_vec([n, MNIST_IMAGE_SIZE], images).unwrap();
    println!("Parsed images");
    let labels = ndarray::Array2::from_shape_fn([n, MNIST_LABELS], |(row, column)| {
        (labels[row] == ((column + 1) % MNIST_LABELS) as u8) as u8 as f32
    });
    println!("Parsed labels");

    // Rate coded
    let images = rate_coding(images, TIME_STEPS);
    println!("Rate coded images");
    let labels = rate_coding(labels, TIME_STEPS);
    println!("Rate coded labels");

    // TODO Need to cache the rate coded data otherwise it takes way too long to re-run this between runs.

    // Column-major
    let images = images
        .axis_chunks_iter(Axis(1), BATCH_SIZE)
        .map(|axis| {
            axis.axis_iter(Axis(0))
                .map(|axis| CudaMatrix::from_ndarray(stream.clone(), axis))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    println!("Loaded images");
    let labels = labels
        .axis_chunks_iter(Axis(1), BATCH_SIZE)
        .map(|axis| {
            axis.axis_iter(Axis(0))
                .map(|axis| CudaMatrix::from_ndarray(stream.clone(), axis))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    println!("Loaded labels");

    (images, labels)
}

#[test]
fn mnist() {
    // Define network hidden layers.
    let hidden_layers = [784, 800, 10];

    // Get VRAM estimate.
    let est = snnsim::cuda::net::Network::vram_estimate(
        MNIST_IMAGE_SIZE,
        &hidden_layers,
        BATCH_SIZE,
        TIME_STEPS,
    );
    println!(
        "estimated vram: {:.2}gb",
        est as f32 / (1024 * 1024 * 1024) as f32
    );

    // Network
    // Inspired from https://en.wikipedia.org/wiki/MNIST_database
    let mut network = snnsim::cuda::net::Network::new(
        MNIST_IMAGE_SIZE,
        &hidden_layers,
        rand_distr::StandardNormal,
        BATCH_SIZE,
        TIME_STEPS,
    );

    // Clone CUDA handles for easier access.
    let stream = network.stream.clone();
    let context = network.context.clone();
    let cublas = network.layers[0].cublas.clone();

    // Get dataset.
    let (train_images, train_labels) = transform_data(get_training().unwrap(), stream.clone());
    println!("Got training data");
    let (test_images, test_labels) = transform_data(get_testing().unwrap(), stream.clone());
    println!("Got testing data");

    // Progress bar style
    let multi_bar = MultiProgress::new();
    let style = ProgressStyle::with_template(
        "{bar:40} {pos:>7}/{len:>7} [{elapsed_precise} / {eta_precise}] {per_sec} {msg}",
    )
    .unwrap();
    let epochs_bar = multi_bar.add(
        ProgressBar::new(EPOCHS as u64)
            .with_style(style.clone())
            .with_message("epochs"),
    );

    // Set time accumulators.
    let mut foreprop_time = Duration::ZERO;
    let mut backprop_time = Duration::ZERO;
    let mut testing_time = Duration::ZERO;

    // Set accuracy trackers.
    let mut best_inaccuracy = f32::NAN;
    let mut testing_inaccuracy;
    let mut training_inaccuracy;

    // Create errors matrix.
    let mut spike_errors = CudaMatrix::zeros(stream.clone(), BATCH_SIZE, MNIST_LABELS).unwrap();

    // Iterate over epochs.
    for epoch in 0..EPOCHS {
        let mut training_errors = 0f32;

        // Train
        for (batch_images, batch_labels) in train_images.iter().zip(train_labels.iter()) {
            // Re-zero `spike_errors` so it can be re-used to calculate training inaccuracy.
            stream.memset_zeros(&mut spike_errors.slice).unwrap();

            // Clear stored gradients.
            network.clear();

            // Start measuring foreprop time.
            stream.synchronize().unwrap();
            let start = Instant::now();

            // Iterate across time steps.
            for (timestep_images, timestep_labels) in batch_images.iter().zip(batch_labels.iter()) {
                let timestep_spikes = network.forward(timestep_images);
                
                // #[cfg(debug_assertions)]
                // {
                //     let ndarray = timestep_spikes.to_ndarray(stream.clone());
                //     assert!(ndarray.iter().all(|f| *f == 0f32 || *f == 1f32));
                // }
                
                // Adds errors from this timestep:
                // `spike_errors += (timestep_labels - timestep_spikes).abs()`
                snnsim::cuda::kernels::abs_diff::run_function(
                    &mut spike_errors,
                    timestep_labels,
                    timestep_spikes,
                    context.clone(),
                    stream.clone(),
                );
            }

            // #[cfg(debug_assertions)]
            // {
            //     let ndarray = spike_errors.to_ndarray(stream.clone());
            //     assert!(ndarray.iter().all(|f| *f <= TIME_STEPS as f32));
            // }

            // Sum all errors from this batch.
            let mut errors = 0f32;
            unsafe {
                cublas
                    .asum(
                        AsumConfig {
                            n: spike_errors.len() as i32,
                            incx: 1,
                        },
                        &spike_errors.slice,
                        &mut errors,
                    )
                    .unwrap()
            };
            debug_assert!(errors <= (spike_errors.len() * TIME_STEPS) as f32);

            // Adds errors from this batch to total errors.
            training_errors += errors;

            // Add foreprop time.
            stream.synchronize().unwrap();
            foreprop_time += start.elapsed();

            // Start measuring backprop time.
            stream.synchronize().unwrap();
            let start = Instant::now();

            // Calculate weight updates.
            let _weight_updates = network.backward(batch_labels).finish();

            // Add backprop time.
            stream.synchronize().unwrap();
            backprop_time += start.elapsed();

            // Update weights.
            network.update(LEARNING_RATE);
        }

        debug_assert!(training_errors <= (spike_errors.len() * train_images.len() * TIME_STEPS) as f32);

        // Calculate percentage inaccurac.
        // `training_errors / (batch size * mnist labels * training batches * time steps)`.
        training_inaccuracy = training_errors / (spike_errors.len() * train_images.len() * TIME_STEPS) as f32;

        // start measuring foreprop time.
        stream.synchronize().unwrap();
        let start = Instant::now();

        let mut testing_errors = 0f32;

        // Test
        for (batch_images, batch_labels) in test_images.iter().zip(test_labels.iter()) {
            // Re-zero `spike_errors` so it can be re-used to calculate testing inaccuracy.
            stream.memset_zeros(&mut spike_errors.slice).unwrap();

            // Clear stored gradients.
            network.clear();

            for (timestep_images, timestep_labels) in batch_images.iter().zip(batch_labels.iter()) {
                // Iterate across time steps.
                // stream.synchronize().unwrap();
                // let foreward_start = Instant::now();
                let timestep_spikes = network.forward(timestep_images);

                // #[cfg(debug_assertions)]
                // {
                //     let ndarray = timestep_spikes.to_ndarray(stream.clone());
                //     assert!(ndarray.iter().all(|f| *f == 0f32 || *f == 1f32));
                // }

                // stream.synchronize().unwrap();
                // println!("foreward_start: {:.2?}",foreward_start.elapsed());
                // let abs_diff = Instant::now();
                snnsim::cuda::kernels::abs_diff::run_function(
                    &mut spike_errors,
                    timestep_labels,
                    timestep_spikes,
                    context.clone(),
                    stream.clone(),
                );
                // stream.synchronize().unwrap();
                // println!("abs_diff: {:.2?}",abs_diff.elapsed());
            }

            // #[cfg(debug_assertions)]
            // {
            //     let ndarray = spike_errors.to_ndarray(stream.clone());
            //     assert!(ndarray.iter().all(|f| *f <= TIME_STEPS as f32));
            // }

            // Sum all inaccurate spikes.
            let mut errors = 0f32;
            unsafe {
                cublas
                    .asum(
                        AsumConfig {
                            n: spike_errors.len() as i32,
                            incx: 1,
                        },
                        &spike_errors.slice,
                        &mut errors,
                    )
                    .unwrap()
            };
            debug_assert!(errors <= (spike_errors.len() * TIME_STEPS) as f32);
            testing_errors += errors;
        }

        debug_assert!(testing_errors <= (spike_errors.len() * test_images.len() * TIME_STEPS) as f32);

        // Calculate percentage inaccuracy.
        testing_inaccuracy = testing_errors / (spike_errors.len() * test_images.len() * TIME_STEPS) as f32;

        // Add testing time.
        stream.synchronize().unwrap();
        testing_time += start.elapsed();

        // Store best testing inaccuracy.
        best_inaccuracy = match best_inaccuracy.partial_cmp(&testing_inaccuracy) {
            Some(Ordering::Greater) => testing_inaccuracy,
            _ => testing_inaccuracy,
        };

        // Update epochs display.
        epochs_bar.set_message(format!(
            "train: {:.2}%, test: {:.2}%, foreprop: {:.3?}, backprop: {:.3?}, testing: {:.3?}",
            training_inaccuracy * 100f32,
            testing_inaccuracy * 100f32,
            foreprop_time.div_f32((1 + epoch) as f32),
            backprop_time.div_f32((1 + epoch) as f32),
            testing_time.div_f32((1 + epoch) as f32)
        ));

        // Increment epoch display.
        epochs_bar.inc(1);
    }
    epochs_bar.finish();
}

const BASE_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/";

pub fn get_training() -> Result<(Vec<Vec<u8>>, Vec<u8>), GetMnistDataError> {
    get_mnist_data("train")
}
pub fn get_testing() -> Result<(Vec<Vec<u8>>, Vec<u8>), GetMnistDataError> {
    get_mnist_data("t10k")
}

#[derive(Debug, Error)]
pub enum GetMnistDataError {
    #[error("Failed to donwload images: {0}")]
    DownloadImages(DownloadAndDecompressError),
    #[error("Failed to parse images: {0}")]
    ParseImages(Box<dyn std::error::Error>),
    #[error("Failed to donwload labels: {0}")]
    DownloadLabels(DownloadAndDecompressError),
    #[error("Failed to parse labels: {0}")]
    ParseLabels(Box<dyn std::error::Error>),
}

fn get_mnist_data(prefix: &str) -> Result<(Vec<Vec<u8>>, Vec<u8>), GetMnistDataError> {
    // Download image data
    let img_data = download_and_decompress(&format!("{}-images-idx3-ubyte.gz", prefix))
        .map_err(GetMnistDataError::DownloadImages)?;
    println!("Got image data");
    let images = parse_idx_images(&img_data).map_err(GetMnistDataError::ParseImages)?;
    println!("Parsed image data");

    // Download label data
    let label_data = download_and_decompress(&format!("{}-labels-idx1-ubyte.gz", prefix))
        .map_err(GetMnistDataError::DownloadLabels)?;
    println!("Got label data");
    let labels = parse_idx_labels(&label_data).map_err(GetMnistDataError::ParseLabels)?;
    println!("Parsed label data");

    Ok((images, labels))
}

#[derive(Debug, Error)]
pub enum DownloadAndDecompressError {
    #[error("Failed to get response: {0}")]
    Get(reqwest::Error),
    #[error("Failed to get bytes: {0}")]
    Bytes(reqwest::Error),
    #[error("Failed to read bytes: {0}")]
    Read(std::io::Error),
}

fn download_and_decompress(filename: &str) -> Result<Vec<u8>, DownloadAndDecompressError> {
    let url = format!("{}{}", BASE_URL, filename);
    let cache = format!("{TMP}/{}", filename);
    let mut decompressed = Vec::new();

    if !std::fs::exists(&cache).unwrap() {
        let bytes = blocking::get(&url)
            .map_err(DownloadAndDecompressError::Get)?
            .bytes()
            .map_err(DownloadAndDecompressError::Bytes)?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&cache)
            .unwrap();
        file.write_all(&bytes).unwrap();
    }
    let file = std::fs::OpenOptions::new().read(true).open(cache).unwrap();
    GzDecoder::new(&file)
        .read_to_end(&mut decompressed)
        .map_err(DownloadAndDecompressError::Read)?;
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
