use ndarray::ArrayBase;
use ndarray::linalg::general_mat_mul as mat_mul;
use ndarray::{Array, Array2, Data, Ix2, RawData, array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{self, Distribution};
use std::f32::consts::PI;
use std::iter::repeat;
use snnsim::encode::rate_coding;
use snnsim::net::Network;

// It seems like when training an SNN to do a digital task. It seems effectively
// very difficult to train the network to train the network to do this task
// effectively due to its inherently recurrent nature.
//
// In this specific example, the inhibitory neuron will always eventually fire
// as a result of the non-linear decay rate, meaning that the network will never
// acheive 100% accuracy and will always be simply balancing how good it is at
// each example.
#[test]
fn xor() {
    let time_steps = 100;

    // Pre-synpatic potentials e.g. the input encodings across time
    let inputs = array![[1f32, 1f32], [0f32,0f32], [1f32,0f32], [0f32,1f32]].into_dyn();
    let targets = array![0f32,0f32,1f32,1f32].into_dyn();
    let rate_encoded_inputs = rate_coding(inputs,100);
    let rate_encoded_targets = rate_coding(targets,100);
    println!("rate_encoded_inputs.shape(): {:?}", rate_encoded_inputs.shape());
    println!("rate_encoded_targets.shape(): {:?}", rate_encoded_targets.shape());
    assert!(false);

    // the shape of our XOR network is 2->3->1
    // the weights will be 2x3 and 3x1
    let weights = vec![
        array![[2f32, 0.21f32, 0f32], [0f32, 0.21f32, 2f32]],
        array![[1f32], [-5f32], [1f32]],
    ];
    let mut net = Network::new(2, &[3, 1], rand_distr::StandardNormal);
    net.weights = weights;


    for epoch in 0..100 {
        let mut total = Array::from_elem((1, 1), 0f32);
        for t in 0..time_steps {
            let s = net.forward(inputs[t].clone());
            // println!("s: {:?}",s.as_slice().unwrap());
            total = total + s;
        }
        total /= time_steps as f32;
        println!("total: {:?}", total.as_slice().unwrap());

        let updates = net.backward(targets.as_slice());
        // println!("updates:");
        // for update in &updates {
        //     println!("\t{:?} {:.5?}",update.shape(), update.as_slice().unwrap());
        // }
        net.update(0.1f32, updates);
    }
}