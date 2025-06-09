use std::sync::Arc;

use cudarc::driver::CudaStream;
use ndarray::Axis;
use ndarray_rand::rand_distr;
use snnsim::{cuda::CudaMatrix, encode::rate_coding};

const MNIST_IMAGE_SIZE: usize = 28 * 28;
const MNIST_LABELS: usize = 10;
const TIME_STEPS: usize = 100;
const BATCH_SIZE: usize = 10_000;
const PIXEL_FIDELITY: f32 = 255f32;
const EPOCHS: usize = 1_000;

fn transform_data(
    (images, labels): (Vec<Vec<u8>>, Vec<u8>),
    stream: Arc<CudaStream>,
) -> (Vec<CudaMatrix>, Vec<CudaMatrix>) {
    let n = images.len();
    assert_eq!(n, labels.len());
    // Row-major
    let images = images
        .into_iter()
        .flatten()
        .map(|x| x as f32 / PIXEL_FIDELITY)
        .collect();
    let images = ndarray::Array2::from_shape_vec([n, MNIST_IMAGE_SIZE], images).unwrap();
    let labels = ndarray::Array2::from_shape_fn([n, MNIST_LABELS], |(row, column)| {
        (labels[row] == ((column + 1) % MNIST_LABELS) as u8) as u8 as f32
    });

    // Rate coded
    let images = rate_coding(images, TIME_STEPS);
    let labels = rate_coding(labels, TIME_STEPS);

    // Column-major
    let images = images
        .axis_iter(Axis(0))
        .map(|axis| CudaMatrix::from_ndarray(stream.clone(), axis))
        .collect::<Vec<_>>();
    let labels = labels
        .axis_iter(Axis(0))
        .map(|axis| CudaMatrix::from_ndarray(stream.clone(), axis))
        .collect::<Vec<_>>();

    (images, labels)
}

#[test]
fn mnist() {
    // Network
    // Inspired from https://en.wikipedia.org/wiki/MNIST_database
    let network = snnsim::cuda::net::Network::new(
        MNIST_IMAGE_SIZE,
        &[784, 800, 10],
        rand_distr::StandardNormal,
        BATCH_SIZE,
        TIME_STEPS,
    );
    let stream = network.stream.clone();

    let (train_images, train_labels) =
        transform_data(snnsim::mnist::get_training().unwrap(), stream.clone());
    let (test_images, test_labels) =
        transform_data(snnsim::mnist::get_testing().unwrap(), stream.clone());

    todo!(
        "batches need to be properly supported and how this is handled in training vs testing needs to be updated"
    );
}
