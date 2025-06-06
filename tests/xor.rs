use std::cmp::Ordering;
use std::time::Duration;
use std::time::Instant;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::Array3;
use ndarray::Axis;
use ndarray::array;
use ndarray_rand::rand_distr::{self};
use snnsim::PollingIterator;
use snnsim::PollingResult;
use snnsim::ZippedPollingIterator;
use snnsim::cuda::CudaMatrix;
use snnsim::encode::rate_coding;
use snnsim::net::Network;

// It seems like when training an SNN to do a digital task. It seems effectively
// very difficult to train the network to train the network to do this task
// effectively due to its inherently recurrent nature.
//
// In this specific example, the inhibitory neuron will always eventually fire
// as a result of the non-linear decay rate, meaning that the network will never
// achieve 100% accuracy and will always be simply balancing how good it is at
// each example.
#[test]
fn xor_cpu() {
    let epochs = 10_000;
    let batch_size = 4;
    let time_steps = 50;
    let learning_rate = 0.01f32;
    let inputs = array![[1f32, 1f32], [0f32, 0f32], [1f32, 0f32], [0f32, 1f32]];
    let targets = array![[0f32], [0f32], [1f32], [1f32]];
    let rate_encoded_inputs = rate_coding(inputs, time_steps);
    let rate_encoded_targets = rate_coding(targets, time_steps);
    let output_spikes = rate_encoded_targets.dim().2;

    // the shape of our XOR network is 2->3->1
    // the weights will be 2x3 and 3x1
    let mut net = Network::new(
        2,
        &[3, output_spikes],
        rand_distr::StandardNormal,
        batch_size,
    );
    // net.weights = vec![
    //     array![[2f32, 0.21f32, 0f32], [0f32, 0.21f32, 2f32]],
    //     array![[1f32], [-5f32], [1f32]],
    // ];

    // Create progress bars for training.
    // Progress bar style
    let multi_bar = MultiProgress::new();
    let style = ProgressStyle::with_template(
        "{bar:40} {pos:>7}/{len:>7} [{elapsed_precise} / {eta_precise}] {per_sec} {msg}",
    )
    .unwrap();
    let epochs_bar = multi_bar.add(
        ProgressBar::new(epochs)
            .with_style(style.clone())
            .with_message("epochs"),
    );
    let time_steps_bar = multi_bar.add(
        ProgressBar::new(time_steps as u64)
            .with_style(style.clone())
            .with_message("time steps"),
    );

    let mut foreprop_time = Duration::ZERO;
    let mut backprop_time = Duration::ZERO;

    let mut best_accuracy = f32::MAX;
    for epoch in 0..epochs {
        // Perform forward pass through all time steps.
        let start = Instant::now();
        let mut spikes = Array3::zeros([0, batch_size, output_spikes]);
        for time_step_inputs in rate_encoded_inputs.axis_iter(Axis(0)) {
            let time_step_spikes = net.forward(time_step_inputs.to_owned());
            spikes
                .append(Axis(0), time_step_spikes.insert_axis(Axis(0)).view())
                .unwrap();
            time_steps_bar.inc(1);
        }
        foreprop_time += start.elapsed();

        // If we use `time_steps_bar.reset()` it resets the `/s` metric, since
        // we want to preserve this we just reset parts.
        time_steps_bar.set_position(0);
        time_steps_bar.reset_elapsed();
        time_steps_bar.reset_eta();

        // Get epoch accuracy.
        let accuracy =
            (spikes - &rate_encoded_targets).abs().sum() / (time_steps as f32 * batch_size as f32);
        best_accuracy = match best_accuracy.partial_cmp(&accuracy) {
            Some(Ordering::Greater) => accuracy,
            _ => best_accuracy,
        };

        // Display accuracy.
        epochs_bar.set_message(format!(
            "current: {:.2}%, best: {:.2}%, backprop: {:?}, foreprop: {:?}",
            accuracy * 100f32,
            best_accuracy * 100f32,
            backprop_time.div_f32((1 + epoch) as f32),
            foreprop_time.div_f32((1 + epoch) as f32)
        ));

        // Calculate weight updates.
        let start = Instant::now();
        let updates = net.backward(&rate_encoded_targets).finish();
        backprop_time += start.elapsed();

        // Update weights.
        net.update(learning_rate, updates);

        // Increment epoch display.
        epochs_bar.inc(1);
    }
    time_steps_bar.finish_and_clear();
    epochs_bar.finish();

    // Display final weights.
    println!("weights:");
    for weights in &net.weights {
        println!(
            "\t{:?} {:.5?}",
            weights.shape(),
            weights.as_slice().unwrap()
        );
    }
    assert!(false);
}

// TODO `xor_cpu_foreprop` and `xor_cuda_foreprop` should be the same test which
// tests the CPU and CUDA versions both work and give the same values.
#[test]
fn foreprop() {
    let batch_size = 4;
    let time_steps = 3;
    let output_spikes = 1;
    let weights = vec![
        array![[2f32, 0.21f32, 0f32], [0f32, 0.21f32, 2f32]],
        array![[1f32], [-5f32], [1f32]],
    ];

    // Setup CPU network
    let mut cpu_net = Network::new(
        2,
        &[3, output_spikes],
        rand_distr::StandardNormal,
        batch_size,
    );
    cpu_net.weights = weights.clone();

    // Setup GPU network
    let mut gpu_net = snnsim::cuda::net::Network::new(
        2,
        &[3, output_spikes],
        rand_distr::StandardNormal,
        batch_size,
        time_steps,
    );
    let stream = gpu_net.stream.clone();
    gpu_net.weights = weights
        .iter()
        .map(|w| CudaMatrix::from_ndarray(stream.clone(), w.view()))
        .collect();

    let inputs = array![[1f32, 1f32], [0f32, 0f32], [1f32, 0f32], [0f32, 1f32]];
    let targets = array![[0f32], [0f32], [1f32], [1f32]];

    let rate_encoded_inputs = rate_coding(inputs, time_steps);
    let rate_encoded_targets = rate_coding(targets, time_steps);

    let cpu_inputs = rate_encoded_inputs.clone();
    let gpu_inputs = rate_encoded_inputs
        .axis_iter(Axis(0))
        .map(|axis| CudaMatrix::from_ndarray(stream.clone(), axis))
        .collect::<Vec<_>>();

    let cpu_targets = rate_encoded_targets.clone();
    let gpu_targets = rate_encoded_targets
        .axis_iter(Axis(0))
        .map(|axis| CudaMatrix::from_ndarray(stream.clone(), axis))
        .collect::<Vec<_>>();

    for (time_step, (cpu_input, gpu_input)) in cpu_inputs
        .axis_iter(Axis(0))
        .zip(gpu_inputs.into_iter())
        .enumerate()
    {
        assert_eq!(cpu_input, gpu_input.to_ndarray(stream.clone()));

        let cpu_spikes = cpu_net.forward(cpu_input.to_owned());
        let gpu_spikes = gpu_net.forward(gpu_input);

        for (cpu_layer, gpu_layer) in cpu_net.layers.iter().zip(gpu_net.layers.iter()) {
            let cpuw = &cpu_layer.weighted_inputs[time_step];
            assert_eq!(
                cpuw,
                gpu_layer.weighted_inputs[time_step].to_ndarray(stream.clone())
            );
            assert_eq!(
                cpu_layer.membrane_potential,
                gpu_layer.membrane_potential.to_ndarray(stream.clone())
            );
        }
        assert_eq!(cpu_spikes, gpu_spikes.to_ndarray(gpu_net.stream.clone()));
    }

    let cpu_back_iter = cpu_net.backward(&cpu_targets);
    let gpu_back_iter = gpu_net.backward(&gpu_targets);
    let mut back_iter = cpu_back_iter.zip(gpu_back_iter);

    let (cpu_errors, gpu_errors) = loop {
        back_iter = match back_iter.next() {
            PollingResult::Complete((x, y)) => break (x, y),
            PollingResult::Incomplete(x) => {
                let ZippedPollingIterator {
                    a: cpu_iter,
                    b: gpu_iter,
                } = &x;

                assert_eq!(
                    cpu_iter.delta_weights,
                    gpu_iter
                        .delta_weights
                        .iter()
                        .map(|dw| dw.to_ndarray(stream.clone()))
                        .collect::<Vec<_>>()
                );

                x
            }
        };
    };

    assert_eq!(
        cpu_errors,
        gpu_errors
            .into_iter()
            .map(|dw| dw.to_ndarray(stream.clone()))
            .collect::<Vec<_>>()
    );
}
