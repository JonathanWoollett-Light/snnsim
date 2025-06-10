use core::f32;
use std::{
    cmp::Ordering,
    sync::Arc,
    time::{Duration, Instant},
};

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
const TIME_STEPS: usize = 30;
const BATCH_SIZE: usize = 10_000;
const PIXEL_FIDELITY: f32 = 255f32;
const EPOCHS: usize = 1_000;
const LEARNING_RATE: f32 = 0.003f32;
const TRAINING_SAMPLES: usize = 60_000;
const TESTING_SAMPLES: usize = 10_000;

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

/// Resets a sub progress bar preserving the `/s` metric.
///
/// If we use `bar.reset()` it resets the `/s` metric, since
/// we want to preserve this we just reset parts.
fn reset_subbar(bar: &ProgressBar) {
    bar.set_position(0);
    bar.reset_elapsed();
    bar.reset_eta();
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
    let foreprop_bar = multi_bar.add(
        ProgressBar::new(BATCH_SIZE as u64)
            .with_style(style.clone())
            .with_message("foreprogation time steps"),
    );
    let backprop_bar = multi_bar.add(
        ProgressBar::new(TIME_STEPS as u64)
            .with_style(style.clone())
            .with_message("backpropagation time steps"),
    );
    let test_bar = multi_bar.add(
        ProgressBar::new(TIME_STEPS as u64)
            .with_style(style.clone())
            .with_message("test time steps"),
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
        // Re-zero `spike_errors` so it can be re-used to calculate training inaccuracy.
        stream.memset_zeros(&mut spike_errors.slice).unwrap();

        let mut training_errors = 0f32;

        // Train
        for (batch_images, batch_labels) in train_images.iter().zip(train_labels.iter()) {
            // Clear stored gradients.
            network.clear();

            // Start measuring foreprop time.
            let start = Instant::now();

            // Iterate across time steps.
            for (timestep_images, timestep_labels) in batch_images.iter().zip(batch_labels.iter()) {
                let timestep_spikes = network.forward(timestep_images);
                snnsim::cuda::kernels::abs_diff::run_function(
                    &mut spike_errors,
                    timestep_labels,
                    timestep_spikes,
                    context.clone(),
                    stream.clone(),
                );
                foreprop_bar.inc(1);
            }

            // Sum all inaccurate spikes.
            let mut errors = 0f32;
            unsafe {
                cublas
                    .asum(
                        AsumConfig {
                            n: BATCH_SIZE as i32,
                            incx: 1,
                        },
                        &spike_errors.slice,
                        &mut errors,
                    )
                    .unwrap()
            };
            training_errors += errors;

            // Add foreprop time.
            foreprop_time += start.elapsed();

            // Reset foreprop bar.
            reset_subbar(&foreprop_bar);

            // Start measuring backprop time.
            let start = Instant::now();

            // Calculate weight updates.
            let weight_updates = network.backward(batch_labels).finish();

            // Add backprop time.
            backprop_time += start.elapsed();

            // Update weights.
            network.update(LEARNING_RATE, &weight_updates);
        }

        // Calculate percentage inaccuracy.
        training_inaccuracy =
            training_errors / (MNIST_LABELS * TIME_STEPS * TRAINING_SAMPLES) as f32;

        // start measuring foreprop time.
        let start = Instant::now();

        // Re-zero `spike_errors` so it can be re-used to calculate testing inaccuracy.
        stream.memset_zeros(&mut spike_errors.slice).unwrap();

        // Test
        for (timestep_images, timestep_labels) in test_images.iter().zip(test_labels.iter()) {
            // Iterate across time steps.
            let timestep_spikes = network.forward(timestep_images);
            snnsim::cuda::kernels::abs_diff::run_function(
                &mut spike_errors,
                timestep_labels,
                timestep_spikes,
                context.clone(),
                stream.clone(),
            );
            test_bar.inc(1);
        }

        // Sum all inaccurate spikes.
        let mut errors = 0f32;
        unsafe {
            cublas
                .asum(
                    AsumConfig {
                        n: BATCH_SIZE as i32,
                        incx: 1,
                    },
                    &spike_errors.slice,
                    &mut errors,
                )
                .unwrap()
        };

        // Calculate percentage inaccuracy.
        testing_inaccuracy = errors / (MNIST_LABELS * TIME_STEPS * TESTING_SAMPLES) as f32;

        // Add testing time.
        testing_time += start.elapsed();

        // Store best testing inaccuracy.
        best_inaccuracy = match best_inaccuracy.partial_cmp(&testing_inaccuracy) {
            Some(Ordering::Greater) => testing_inaccuracy,
            _ => testing_inaccuracy,
        };

        // Reset foreprop bar.
        reset_subbar(&test_bar);

        // Update epochs display.
        epochs_bar.set_message(format!(
            "train: {:.2}%, test: {:.2}%, foreprop: {:?}, backprop: {:?}, testing: {:?}",
            training_inaccuracy * 100f32,
            testing_inaccuracy * 100f32,
            foreprop_time.div_f32((1 + epoch) as f32),
            backprop_time.div_f32((1 + epoch) as f32),
            testing_time.div_f32((1 + epoch) as f32)
        ));

        // Increment epoch display.
        epochs_bar.inc(1);
    }
    foreprop_bar.finish_and_clear();
    backprop_bar.finish_and_clear();
    test_bar.finish_and_clear();
    epochs_bar.finish();
}
