use std::sync::Arc;

use cudarc::driver::CudaStream;
use ndarray::Axis;
use ndarray_rand::rand_distr;
use snnsim::{PollingIterator, cuda::CudaMatrix, encode::rate_coding};

const MNIST_IMAGE_SIZE: usize = 28 * 28;
const MNIST_LABELS: usize = 10;
const TIME_STEPS: usize = 100;
const BATCH_SIZE: usize = 10_000;
const PIXEL_FIDELITY: f32 = 255f32;
const EPOCHS: usize = 1_000;
const LEARNING_RATE: f32 = 0.003f32;

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
    let labels = ndarray::Array2::from_shape_fn([n, MNIST_LABELS], |(row, column)| {
        (labels[row] == ((column + 1) % MNIST_LABELS) as u8) as u8 as f32
    });

    // Rate coded
    let images = rate_coding(images, TIME_STEPS);
    let labels = rate_coding(labels, TIME_STEPS);

    // Column-major
    let images = images
        .axis_chunks_iter(Axis(1), BATCH_SIZE)
        .map(|axis| {
            axis.axis_iter(Axis(0))
                .map(|axis| CudaMatrix::from_ndarray(stream.clone(), axis))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let labels = labels
        .axis_chunks_iter(Axis(1), BATCH_SIZE)
        .map(|axis| {
            axis.axis_iter(Axis(0))
                .map(|axis| CudaMatrix::from_ndarray(stream.clone(), axis))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    (images, labels)
}

#[test]
fn mnist() {
    // Network
    // Inspired from https://en.wikipedia.org/wiki/MNIST_database
    let mut network = snnsim::cuda::net::Network::new(
        MNIST_IMAGE_SIZE,
        &[784, 800, 10],
        rand_distr::StandardNormal,
        BATCH_SIZE,
        TIME_STEPS,
    );
    let stream = network.stream.clone();
    let context = network.context.clone();

    let (train_images, train_labels) =
        transform_data(snnsim::mnist::get_training().unwrap(), stream.clone());
    let (test_images, test_labels) =
        transform_data(snnsim::mnist::get_testing().unwrap(), stream.clone());
    let [test_images] = test_images.as_slice() else {
        unreachable!()
    };
    let [test_labels] = test_labels.as_slice() else {
        unreachable!()
    };

    let mut spike_errors = CudaMatrix::zeros(stream.clone(), BATCH_SIZE, MNIST_LABELS);
    for epoch in 0..EPOCHS {
        // Train
        stream.memset_zeros(&mut spike_errors.slice).unwrap();
        for (batch_images, batch_labels) in train_images.iter().zip(train_labels.iter()) {
            // Iterate across time steps.
            for (timestep_images,timestep_labels) in batch_images.iter().zip(batch_labels.iter()) {
                let timestep_spikes = network.forward(timestep_images);
                snnsim::cuda::kernels::abs_diff::run_function(
                    &mut spike_errors,
                    timestep_labels, 
                    timestep_spikes,
                    context.clone(),
                    stream.clone()
                );
            }
            // Calculate weight updates.
            let weight_updates = network.backward(batch_labels).finish();

            // Update weights.
            network.update(LEARNING_RATE, &weight_updates);
        }
        // Test
        stream.memset_zeros(&mut spike_errors.slice).unwrap();
        for (timestep_images,timestep_labels) in test_images.iter().zip(test_labels.iter()) {
            // Iterate across time steps.
            let timestep_spikes = network.forward(timestep_images);
            snnsim::cuda::kernels::abs_diff::run_function(
                &mut spike_errors,
                timestep_labels, 
                timestep_spikes,
                context.clone(),
                stream.clone()
            );
        }
    }
}
