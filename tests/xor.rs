use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::Array3;
use ndarray::Axis;
use ndarray::array;
use ndarray_rand::rand_distr::{self};
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
fn xor() {
    let epochs = 4000;
    let batch_size = 4;
    let time_steps = 200;
    let inputs = array![[1f32, 1f32], [0f32, 0f32], [1f32, 0f32], [0f32, 1f32]];
    let targets = array![[0f32], [0f32], [1f32], [1f32]];
    let rate_encoded_inputs = rate_coding(inputs, time_steps);
    let rate_encoded_targets = rate_coding(targets, time_steps);
    let output_spikes = rate_encoded_targets.dim().2;
    println!(
        "rate_encoded_inputs.shape(): {:?}",
        rate_encoded_inputs.shape()
    );
    println!(
        "rate_encoded_targets.shape(): {:?}",
        rate_encoded_targets.shape()
    );

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

    // Progress bar style
    let style = ProgressStyle::with_template(
        "{bar:40} {pos:>7}/{len:>7} [{elapsed_precise} / {eta_precise}] {per_sec} {msg}",
    )
    .unwrap();
    let multibar = MultiProgress::new();
    let epochs_bar = multibar.add(
        ProgressBar::new(epochs)
            .with_style(style.clone())
            .with_message("epochs"),
    );
    let time_steps_bar = multibar.add(
        ProgressBar::new(time_steps as u64)
            .with_style(style.clone())
            .with_message("time steps"),
    );

    for _epoch in 0..epochs {
        let mut spikes = Array3::zeros([0, batch_size, output_spikes]);
        for time_step_inputs in rate_encoded_inputs.axis_iter(Axis(0)) {
            let time_step_spikes = net.forward(time_step_inputs.to_owned());
            spikes
                .append(Axis(0), time_step_spikes.insert_axis(Axis(0)).view())
                .unwrap();
            time_steps_bar.inc(1);
        }
        time_steps_bar.reset();
        let cost =
            (spikes - &rate_encoded_targets).abs().sum() / (time_steps as f32 * batch_size as f32);

        epochs_bar.set_message(format!("cost: {cost}"));
        let updates = net.backward(&rate_encoded_targets);
        net.update(0.001f32, updates);
        epochs_bar.inc(1);
    }
    time_steps_bar.finish_and_clear();
    epochs_bar.finish();

    println!("updates:");
    for update in &net.weights {
        println!("\t{:?} {:.5?}", update.shape(), update.as_slice().unwrap());
    }
    assert!(false);
}
